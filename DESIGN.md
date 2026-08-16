# silent-data-defects — desenho técnico

> Perfilar não é o produto. O produto é **ranquear defeitos de dados pelo quanto mudam a decisão**, e emitir o contrato que impede o defeito de voltar.

Nomes alternativos, mesma tese: `defect-to-decision`, `data-contract-under-scale`.

---

## 1. Tese

Três afirmações que o projeto precisa sustentar com número, não com prosa:

1. **Prevalência não é severidade.** Uma coluna com 30% de nulos e AUC univariada 0,51 é ruído. Uma com 2% de sentinelas e AUC 0,78 muda decisão. Todo relatório de qualidade existente ordena pela primeira métrica.
2. **O contrato de validação deve sair do perfil, não da cabeça do analista.** É o passo que o Great Expectations deixa manual — e é onde o projeto morre na vida real.
3. **Escala é uma decisão de arquitetura, não um adjetivo.** Três camadas de leitura, uma passada de dados, sketches mergeáveis. Sem `pd.read_parquet` do arquivo inteiro em lugar nenhum.

Não-objetivo explícito: relatório HTML com 200 métricas por coluna. Isso já existe (`ydata-profiling`) e não é o valor.

---

## 2. Arquitetura: três camadas de leitura

A diferença entre 40 segundos e "não termina" está em ler o mínimo possível.

### Camada 0 — metadados, zero linhas lidas

O footer do Parquet já entrega, por row-group e por coluna: contagem de linhas, `null_count`, `min`/`max` estatísticos, `distinct_count` (quando o writer gravou), tamanho comprimido. Em um arquivo de 50 GB isso sai em milissegundos.

O que já dá pra reportar sem ler um byte de dado:
- taxa de nulos exata por coluna;
- faixa de valores — e portanto **sentinela óbvia no extremo** (`min = -999` numa coluna de renda);
- colunas constantes (`min == max`);
- desalinhamento de schema entre arquivos do mesmo diretório.

Ninguém faz isso, e é o gesto de engenharia mais barato do repo. Vale um modo `--metadata-only` que roda em qualquer volume.

```python
import pyarrow.parquet as pq
md = pq.ParquetFile(path).metadata
for rg in range(md.num_row_groups):
    for c in range(md.num_columns):
        st = md.row_group(rg).column(c).statistics   # null_count, min, max, distinct_count
```

Para CSV não existe camada 0 — o modo degrada para camada 1 e o README diz isso.

### Camada 1 — uma passada, agregados vetorizados

Motor: **DuckDB**. Faz projection pushdown, lê Parquet e CSV, roda out-of-core, e já tem `approx_count_distinct` (HLL) e `approx_quantile` (t-digest) nativos.

Decisão de engenharia importante: **não implementar HLL/KLL na mão.** O valor do projeto não é reimplementar sketch — é o que se faz com o resultado. Se algum sketch faltar no DuckDB, usar `datasketches` (Apache DataSketches, mesma implementação do Deequ/Druid), não escrever do zero.

> **Corrigido pela medição (marco 1).** O parágrafo abaixo dizia "empacote tudo numa query
> só, uma passada". Medido, isso está certo para agregado barato e errado para agregado
> pesado: 400 `count(*) FILTER` numa query custam 16,7 s contra 1,8 s em lotes de 16, e 24
> `approx_top_k` custam 11,7 s contra 1,1 s em lotes de 2 — enquanto 104 agregados simples
> ficam *mais rápidos* numa query só (1,1 s contra 2,4 s). O lote é por **classe de custo da
> expressão**, não por coluna, e "número de passadas" não é a métrica de custo. Ver o README
> e `bench/batching.py`.

Erro clássico a evitar: uma query por coluna = k passadas na base. O correto é **uma query com todos os agregados de um lote de colunas**, em blocos de ~50 colunas para não estourar o planner:

```sql
SELECT
  count(*)                                   AS n,
  count(renda)                               AS renda_nonnull,
  approx_count_distinct(renda)               AS renda_ndv,
  approx_quantile(renda, [0.01,0.25,0.5,0.75,0.99]) AS renda_q,
  min(renda), max(renda), avg(renda), stddev_pop(renda),
  -- ... x50 colunas
FROM read_parquet('data/*.parquet');
```

Top-K por coluna sai em passada própria por bloco (`GROUP BY col ORDER BY count DESC LIMIT 256`) — é o insumo do detector de sentinela.

### Camada 2 — sob demanda, só nas colunas suspeitas

Missingness cruzada, associação com alvo, teste de gap, correlação temporal. Roda **apenas nas colunas que a camada 1 marcou**, tipicamente 5–15 de 200. É o que mantém o custo sublinear no número de colunas.

> **Ampliado pela medição (marco 1).** A camada 2 foi desenhada para conter custo. Ela é
> também onde se compra **acurácia** de volta, e isso não estava previsto. `approx_quantile`
> é não determinístico com threads > 1, e o erro se concentra no quantil vizinho a uma massa
> pontual — exatamente onde o teste de gap do D1 lê P01/P99 e o IQR. O gap test portanto não
> usa quantil aproximado: usa `quantile_cont` exato nas poucas colunas com candidato (0,16 s
> por coluna, contra 13 s para a base inteira). Ver `src/sdd/scan/exact.py`.
>
> Pelo mesmo motivo, cardinalidade deixou de ser aproximada por padrão: o HLL do DuckDB errou
> até 17% nesta base, sem parâmetro de precisão, enquanto `count(DISTINCT)` exato custa 1,3 s
> para 26 colunas. O sketch fica só onde a tabela hash não caberia.

### Estado mergeável

Todo agregado tem que ser um monoide (combinável entre partições, sem ordem): contagens somam, min/max combinam, HLL/KLL têm `merge`, Space-Saving tem merge aproximado. Isso é o que permite paralelizar por row-group e, depois, comparar dois lotes sem reler nenhum dos dois. **É também o que permite o drift check ser barato**: guarda-se o sketch no contrato, não os dados.

---

## 3. Catálogo de detectores

Cada detector emite: `coluna`, `tipo`, `massa afetada`, `evidência`, `confiança`. A severidade **não** vem daqui — vem da seção 4.

### D1 · Sentinela numérica disfarçada de valor

`-999`, `9999`, `0`, `-1`, `999999999`. Passam por qualquer validador de tipo e envenenam média, escala e corte de árvore.

Algoritmo, para cada candidato `v` no top-K com massa `m = freq(v)/n`:

1. `m > 0,1%` — abaixo disso não move nada;
2. **teste de gap** — seja `v' ` o valor mais próximo de `v` no suporte, do lado interno. Sentinela é um ponto *isolado*: ~~`|v − v'| > 3·IQR`~~. Renda com valores `0` e depois nada até `800` → o `0` é código de ausente. Renda com `0, 12, 40, 95, …` → `0` é valor legítimo. **Esse teste de gap é o que separa este detector de um heurístico ingênuo, e é o achado central do repo.**

   > **Corrigido pela medição (marco 2).** A intuição do teste estava certa; a escala estava
   > errada. Com `3·IQR`, a armadilha `qtd_produtos = 0` pontua **1,00** contra **0,46** da
   > sentinela verdadeira `idade = 0` — o teste inverte o veredito. A escala correta depende
   > de a coluna ser discreta ou contínua: **espaçamento mediano entre valores adjacentes**
   > numa, **amplitude do suporte restante** (com o lift `massa × ndv` como condição
   > necessária) na outra. Coincide com a divisão de rotas que a camada 1 já fazia por custo.
   > Separação medida: 18x na rota discreta, 1.104x na contínua.
3. `v` casa com padrão canônico (`±1, ±9, ±99, ±999, ±9999, 0, 99999999`) → reforça a confiança, mas **não é condição necessária** (não pode ser só uma lista de constantes, senão é regex disfarçada de detector);
4. `v` é exatamente o `min` ou `max` da coluna → reforça.

Confiança = combinação de (2), (3), (4). Reportar (2) como evidência primária.

### D2 · Sentinela textual e temporal

Match exato contra `{"", " ", "N/A", "n/a", "NA", "null", "NULL", "None", "-", "?", "#N/A", "NaN", "desconhecido", "não informado"}` e, em datas, `1900-01-01`, `1970-01-01` (epoch usado como ausente), `9999-12-31`, `0001-01-01`. Barato e de alto rendimento — mas é o detector menos interessante; não pode ser o carro-chefe.

### D3 · Nulo com estrutura (missingness não aleatória)

O nulo raramente é aleatório. Três cortes, todos `GROUP BY` simples sobre a máscara binária `is_null`:

- **por tempo** — taxa de nulo por bucket de data. `score_bureau` com 2% de nulos até 2024-03-15 e 100% depois é um pipeline quebrado que ninguém viu. **Provavelmente o achado de maior valor prático do projeto inteiro, e é o mais barato de calcular.**
- **entre colunas** — colunas cujos nulos coincidem vêm da mesma origem/join que falhou. Correlação φ sobre a máscara.
- **contra o alvo** — se a taxa de evento difere entre nulo e não-nulo, o nulo *é* sinal e imputar destrói informação. Reportar como "imputar aqui apaga sinal", não como "12% de nulos".

### D4 · Vazamento

Sem treinar modelo nenhum:

- **numérica × alvo binário** — AUC univariada, que é a estatística U de Mann-Whitney e sai direto dos ranks: `AUC = (U) / (n₊·n₋)`. `AUC > 0,95` univariada quase nunca é feature boa; é a resposta vazando.
- **categórica × alvo** — nível com taxa de evento ~100% e suporte não-trivial; informação mútua normalizada.
- **temporal** — coluna de data cujo valor é posterior à data do rótulo em fração relevante das linhas. Exige que o usuário declare qual coluna é a data de referência.

### D5 · Skew entre lotes / treino-produção

Mesma coluna, dois arquivos. PSI e KS calculados **a partir dos sketches KLL**, sem reler dado. Generaliza o CSI por feature.

Nota herdada do `credit-risk-decision-engine`: PSI agregado sobre o escore é pouco sensível — sob safra deteriorada a aprovação caiu 7 p.p. com PSI em 0,027. O default aqui é **por feature**, e agregado só como complemento.

### D6 · Cardinalidade patológica

Categórica com dezenas de milhares de níveis (via HLL), ou nível novo em produção ausente do top-K de treino. Custo real: one-hot que derruba o pipeline, ou target encoding que vaza.

---

## 4. Impacto na decisão — três níveis de evidência

Aqui é onde o projeto ganha ou perde. Resposta honesta primeiro:

> Para afirmar "corrigir a coluna X muda a decisão", é preciso a função de X para decisão. Essa função é o modelo. Não existe atalho que dê o número exato sem ela.

O que existe é uma escada de evidência, e cada degrau precisa ser rotulado no relatório pelo que ele realmente mede. Fingir que o nível A é o nível C é exatamente o tipo de desonestidade que o portfólio não pode ter.

### Nível A — heurística de priorização (sempre disponível, sem alvo, sem modelo)

```
impacto ≈ massa_afetada × alavancagem_univariada
```

onde `alavancagem = 2·|AUC − 0,5|` para alvo binário, ou dispersão relativa quando não há alvo. Serve **só para ordenar a lista de defeitos**, e o relatório precisa dizer isso com essas palavras. Não é percentual de decisão alterada.

### Nível B — regra de decisão declarada (model-free, exato no que mede)

O usuário declara a decisão, que é sempre uma dessas três formas:

```yaml
decision:
  kind: threshold        # aprovar se score >= 0.70
  column: score
  cutoff: 0.70
# ou
  kind: top_k            # revisar os 5% mais suspeitos  (fraud-triage-under-capacity)
  column: score
  fraction: 0.05
# ou
  kind: bucket           # faixas de política por decil
  column: score
  edges: [0.1, 0.3, 0.7]
```

Com a regra dada, mede-se **cruzamento de fronteira**: repara-se o defeito na coluna, recalcula-se o percentil do registro, conta-se quantos atravessam a fronteira declarada. Saída: *"tratar `-999` como ausente move 3,4% dos registros do decil 2 para o decil 7"*. Exato, sem modelo, e limitado a uma coluna por vez — limitação que deve estar escrita.

### Nível C — modelo plugado (número verdadeiro)

Interface opcional: o usuário passa um callable `predict(DataFrame) -> array` ou um artefato `.pkl`/`.txt` (Booster do LightGBM). Aí o número é o de verdade — **taxa de inversão de decisão**:

```
flip_rate = P[ decisão(predict(X_reparado)) ≠ decisão(predict(X_bruto)) ]
```

E, se houver rótulo, o delta de lucro pela mesma função de custo do `credit-risk-decision-engine`. Este é o modo que fecha o arco do portfólio: o defeito de dado vira reais.

**Regra de relatório:** cada linha da tabela de defeitos carrega a etiqueta do nível (A/B/C) que produziu sua severidade. Sem etiqueta, o número mente.

---

## 5. O contrato emitido

Formato: YAML versionado, legível e revisável em PR. Duas seções — o que foi *observado* e o que será *checado*.

Decisão de projeto que carrega o peso: **a tolerância vem do tamanho da amostra, não de constante mágica.** Um `±10%` chutado é o motivo de metade das suítes de validação serem ignoradas por ruído. Com `n = 5.123.400` e taxa de nulo 12,03%, o intervalo de Wilson a 95% é ±0,03 p.p. — então um lote a 12,5% é mudança real, não flutuação, e o alerta é confiável. Com `n = 800`, a mesma taxa tolera ±2,3 p.p. e o alerta correspondente seria ruído.

```yaml
contract_version: 1
source:
  rows: 5123400
  profiled_at: 2026-08-16T10:00:00Z
  engine: duckdb 1.x
  scan: full            # full | sample:0.1 | metadata_only
  decision_ref: {kind: threshold, column: score, cutoff: 0.70}

columns:
  renda:
    dtype: double
    null_rate:
      observed: 0.1203
      tolerance: [0.1195, 0.1211]
      basis: wilson_95 (n=5123400)
    range: {min: 0.0, max: 98000.0, basis: parquet_statistics}
    quantiles: {p01: 900.0, p50: 3100.0, p99: 42000.0, sketch: kll_b64:...}
    findings:
      - detector: D1_sentinela_numerica
        value: -999
        mass: 0.031
        evidence: gap_below_p01 (proximo_valor=380, IQR=2100)
        confidence: 0.93
        impact:
          level: B
          metric: boundary_crossing_rate
          value: 0.034
          statement: "tratar -999 como ausente move 3,4% dos registros atraves do corte 0.70"
        severity: high

checks:
  - id: renda.null_rate
    expr: "null_rate(renda) BETWEEN 0.1195 AND 0.1211"
    severity: high
    rationale: "derivado de impacto nivel B = 3.4% de cruzamento de fronteira"
  - id: renda.no_sentinel
    expr: "count(renda = -999) = 0"
    severity: high
  - id: renda.drift
    expr: "psi_from_kll(renda) < 0.10"
    severity: medium
```

`severity` é **derivada do impacto**, não do tipo de check. É o que diferencia o contrato gerado de uma suíte genérica.

Runner: `validate` lê o contrato, roda os `checks` no lote novo em uma passada, sai com código ≠ 0 e um diff legível. É isso que permite plugar em CI/Airflow — e é o que transforma o projeto de "relatório bonito" em ferramenta.

---

## 6. Avaliação — a parte que faz virar projeto de ciência de dados

Sem isto o repo é tooling. Com isto, é o mesmo padrão do `regulatory-rag-eval`.

**Benchmark de corrupção injetada.** Pega-se uma base limpa, injeta-se um catálogo de N defeitos conhecidos (sentinelas em colunas sorteadas, nulos correlacionados com data, vazamento sintético, drift controlado de magnitude conhecida), roda-se o detector e reporta-se **precisão e recall por tipo de detector**, mais curva de detecção por massa do defeito:

| detector | recall @ massa≥1% | recall @ massa 0,1–1% | precisão | falsos positivos / 100 col |
|---|---|---|---|---|
| D1 sentinela numérica | | | | |
| D3 nulo com estrutura | | | | |
| D4 vazamento | | | | |

Duas perguntas que o benchmark responde e que nenhum concorrente responde:
- **qual a massa mínima de defeito detectável?** (abaixo de que % o detector é cego)
- **quantos falsos positivos por 100 colunas limpas?** — o número que decide se alguém usa a ferramenta ou desliga o alerta.

Aposta sobre o que o benchmark vai mostrar (registrada **antes** de rodar): o detector de sentinela vai ter recall alto e precisão baixa em colunas de contagem, onde `0` legítimo e `0` ausente são indistinguíveis sem semântica. Se acontecer, é achado, não fracasso — e a resposta é o teste de gap ponderado, não uma lista maior de constantes.

> **Resultado (marco 2).** Precisão saiu 100%, não baixa — mas a aposta acertou o lugar do
> problema. `qtd_produtos = 0` é exatamente onde o detector quase falhou, e com a escala
> especificada neste documento (IQR) ele *teria* falhado: a armadilha pontua mais alto que
> a sentinela verdadeira. O que salvou não foi ponderação e sim trocar a escala por
> espaçamento do suporte. A parte da aposta que continua de pé: `0` legítimo e `0` ausente
> **são** indistinguíveis sem semântica quando não há gap — por isso `valor_recuperado = 0`
> sai como distribuição mista, e não como sentinela nem como falso positivo.

---

## 7. Dados

- **Lending Club** (~2,2 M empréstimos, ~150 colunas) — domínio financeiro, escala real, e conhecidamente cheio de vazamento (`total_pymnt`, `recoveries`, `last_pymnt_amnt` só existem depois do desfecho) e de sentinelas. É o caso de demonstração: o detector D4 tem que reencontrar sozinho os vazamentos que a literatura já documentou. **Validação externa gratuita.**
- **Corrupção injetada** sobre uma base limpa — é o que produz os números da seção 6.
- Escala sintética: replicar para 50–100 M de linhas para o benchmark de throughput, deixando claro que é replicação e não dado novo.

---

## 8. Estrutura e marcos

```
silent-data-defects/
├─ README.md
├─ src/sdd/
│  ├─ scan/        metadata.py  duckdb_pass.py  sketches.py
│  ├─ detect/      sentinels.py  missingness.py  leakage.py  drift.py  cardinality.py
│  ├─ impact/      heuristic.py (A)  boundary.py (B)  model.py (C)
│  ├─ contract/    emit.py  validate.py  schema.py
│  └─ cli.py       profile | emit | validate
├─ bench/          inject.py  evaluate.py  throughput.py
└─ reports/
```

Marcos, em ordem de risco decrescente — cada um entrega algo funcionando:

1. ~~**Camada 0 + camada 1 em DuckDB, CLI `profile`.** Prova a escala. Mede o tempo contra `ydata-profiling` no mesmo arquivo (a comparação provavelmente é "termina × não termina" — reportar honestamente como tal, com o limite de memória declarado).~~ **Concluído.**
   A previsão de "termina × não termina" estava errada: o ydata-profiling terminou 5 M × 26
   em 118,9 s com pico de 6,2 GB, contra 4,8 s e 261 MB do sdd — 25x no tempo, 24x na memória.
   O resultado que importa não é o tempo e sim a **memória plana** (209 → 261 MB enquanto o
   dado cresce 50x), porque é ela que decide o comportamento quando a base não cabe na RAM.
   Os três achados de engenharia estão no README.
2. ~~**D1 + D3 + benchmark de injeção.** É o coração; se o recall/precisão não fecharem, é aqui que se descobre, antes de investir no resto.~~ **Concluído.**
   D0/D1/D2/D3 com 100% de precisão e recall sobre 10 defeitos, 0 falsos positivos em
   13 colunas limpas, e a armadilha rejeitada. A aposta registrada abaixo estava **certa
   no diagnóstico e errada na causa**: o detector de sentinela de fato erra em coluna de
   contagem, mas não por massa e sim por **escala** — com IQR a armadilha `qtd_produtos = 0`
   pontua 1,00 contra 0,46 da sentinela real `idade = 0`, invertendo o veredito. A correção
   foi usar espaçamento do suporte em coluna discreta e amplitude em contínua. Ver o README.
3. ~~**`emit` + `validate` com tolerância de Wilson.** O contrato fecha o loop.~~ **Concluído.**
   Com uma correção de rumo: a tolerância **não** pode ser o intervalo de Wilson da emissão.
   Com 5 M linhas ele tem meia-largura de ±0,03 p.p., e ruído amostral puro o viola em lote
   de qualquer tamanho — 1 a 2 alarmes falsos por execução sobre amostra limpa, sempre nas
   colunas que de fato têm nulo. Wilson ficou como documentação do observado; o critério é
   teste de duas proporções com o n do lote novo, mais Bonferroni e piso de efeito prático.
   Ver o README.
4. ~~**Impacto níveis A e B.** A tabela de defeitos passa a ser ordenada por consequência.~~ **Concluído.**
   As duas ordenações de fato divergem: `utilizacao = -1` sobe da 7ª para a 3ª posição (lift
   8,16x na margem) e `score = 0.001` cai da 4ª para a última, com 4,88% de massa e **zero**
   decisões invertidas. Uma correção de rumo: o nível B2 precisou de um defeito injetado
   *na margem* para provar que discrimina — todos os defeitos originais da base tinham
   lift ≈ 1,00, ou seja, exposição igual a massa/10, e a métrica não teria evidência
   nenhuma a favor. Ver o README.
5. **D4, D5, D6 + nível C** com o modelo do `credit-risk-decision-engine` plugado. **Não implementado.**
   O gabarito já contém os três defeitos correspondentes (`valor_recuperado` e
   `n_acionamentos_cobranca` como vazamento, `cep` como cardinalidade patológica) e o
   benchmark os reporta como não encontrados, com o detector marcado como ausente — não
   como acerto. O sinal do D4 inclusive já aparece de graça no nível A: as duas colunas de
   vazamento saem com AUC univariada exatamente 1,0000. Falta o detector próprio, que
   precisa também do caso temporal (coluna cuja data é posterior ao rótulo).

Ambiente: venv própria com `--system-site-packages` (padrão dos outros repos). Dependências novas: `duckdb`, `pyarrow`. Opcional: `datasketches`. **Não** entram: `pandas` no caminho de escala (só no relatório final, sobre dezenas de linhas), Spark, conectores de banco.

---

## 9. Não-objetivos (escrever no README)

- Suporte a "qualquer fonte" — um formato colunar bem feito vale mais que cinco conectores meia-boca.
- Relatório HTML de 200 métricas por coluna.
- Correção automática do dado. A ferramenta detecta, quantifica e contrata. Quem repara é o dono do pipeline.
- Substituir Great Expectations. Ele executa contratos; este projeto **os escreve**.
