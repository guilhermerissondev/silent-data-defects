# silent-data-defects

Perfilar não é o produto. O produto é **ranquear defeitos de dados pelo quanto mudam a decisão**, e emitir o contrato de validação que impede o defeito de voltar.

Ferramenta de qualidade de dados ordena por prevalência: *"a coluna `renda` tem 12% de nulos"*. Isso é ruído — ninguém sabe o que fazer com um relatório de 200 métricas por coluna. O que muda o trabalho de quem usa o dado é *"tratar `-999` como ausente move 3,4% das contas através do corte de aprovação"*.

**Estado: marcos 1 a 4 de 5 concluídos** — camadas de leitura, detectores D0/D1/D2/D3, contrato emitido e executável, impacto na decisão nos níveis A e B (`profile`, `detect`, `emit`, `validate`, `impact`). Falta o marco 5: D4/D5/D6 e o nível C com modelo plugado. Ver [DESIGN.md](DESIGN.md).

---

## Arquitetura: três camadas de leitura

| camada | o que lê | custo medido (5 M × 26 col, 207 MB) |
|---|---|---|
| **0** — footer do Parquet | 32 KB, **zero linhas** | **3 ms** |
| **1** — agregados vetorizados | uma varredura por classe de expressão | **5,1 s** |
| **2** — sob demanda | só as colunas marcadas pela camada 1 | 0,16 s por coluna |

A camada 0 lê apenas o footer — 0,015% do arquivo — e já entrega taxa de nulos exata, min/max, colunas constantes e schema desalinhado entre arquivos. Num arquivo de dezenas de GB o custo continua sendo milissegundos. Nesta base ela sozinha encontrou:

```
[sentinela_no_extremo  ] renda                   min = -999.0
[sentinela_no_extremo  ] dias_desde_ultimo_pgto  max = 9999
[coluna_constante      ] versao_modelo           min = max = 'v3.1'
```

CSV não tem camada 0. O scan degrada para a camada 1 e o relatório diz isso.

---

## Comparação: sdd × ydata-profiling

Mesma base, mesma máquina, nenhum dos dois renderizando HTML — só o cálculo do perfil. O ydata roda em `minimal=True`, que **desliga a matriz de correlação**: é a configuração que o favorece.

| linhas | sdd camada 0 | sdd completo | ydata minimal | pico sdd | pico ydata |
|---:|---:|---:|---:|---:|---:|
| 100 000 | 0,46 s | 0,85 s | 16,77 s | 209 MB | 3 674 MB |
| 1 000 000 | 0,44 s | 1,25 s | 36,16 s | 221 MB | 4 297 MB |
| 5 000 000 | 0,40 s | 4,80 s | 118,90 s | 261 MB | 6 196 MB |

**25x no tempo, 24x na memória.** O número que importa não é o tempo — é a segunda coluna de memória: o sdd fica **plano** (209 → 261 MB enquanto o dado cresce 50x) porque nada é materializado; o ydata começa em 3,7 GB com 100 mil linhas, o que mostra que o custo dele é dominado por estrutura por coluna, não por volume.

Correção de uma previsão errada: o DESIGN original apostava que a comparação seria *"termina × não termina"*. Não foi — o ydata terminou 5 M em 2 minutos. A vantagem real é a memória plana, e é isso que decide quando a base não cabe na RAM.

`python bench/compare_ydata.py --sizes 100000 1000000 5000000`

---

## Detecção: precisão, recall e margem

Base sintética de 5 M linhas com **14 defeitos injetados, 1 armadilha e 12 colunas limpas**, gabarito em manifesto versionado junto com os dados.

| detector | o que faz | TP | FN | FP | precisão | recall |
|---|---|---:|---:|---:|---:|---:|
| **D0** | constante / 100% nula / schema divergente (camada 0) | 1 | 0 | 0 | 100% | 100% |
| **D1** | sentinela numérica pelo teste de gap | 6 | 0 | 0 | 100% | 100% |
| **D2** | código textual e temporal de ausência | 2 | 0 | 0 | 100% | 100% |
| **D3** | nulo com estrutura (tempo, categoria, entre colunas, alvo) | 2 | 0 | 0 | 100% | 100% |
| D4 / D6 | vazamento, cardinalidade | — | — | — | não implementados (marco 5) | |

**0,0 falsos positivos por 100 colunas limpas.**

Precisão 100% sozinha não significa nada — pode ser sorte de calibração. O número que decide é a **margem**:

```
coluna                    valor    massa      rota      razão   limiar  veredito
utilizacao                   -1    1,98%  contínua      1,002    0,005  SENTINELA
renda                      -999    3,10%  contínua    0,01456    0,005  SENTINELA
score                     0,001    4,88%  contínua  1,002e-05    0,005  massa pontual
valor_recuperado              0   72,21%  contínua   1,12e-07    0,005  massa pontual
dias_desde_ultimo_pgto     9999    0,80%  discreta      9.910        8  SENTINELA
idade                         0    1,20%  discreta         18        8  SENTINELA
qtd_produtos                  0   26,16%  discreta          1        8  rejeitado  <- ARMADILHA
```

- rota discreta: positivo mais fraco **18**, negativo mais forte **1** → separação de **18x**
- rota contínua: positivo mais fraco **0,0146**, negativo mais forte **0,00001** → separação de **1.453x**

A separação é confortável nas duas rotas. A **folga até o limiar não é**: 2,2x na discreta e 2,9x na contínua. Uma sentinela um pouco menos isolada passaria despercebida, e isso está registrado como limite conhecido — não como resultado.

`python bench/evaluate.py data/contas.parquet`

---

## O contrato: emitir e executar

`emit` transforma o perfil em contrato; `validate` roda o contrato num lote novo e sai com código ≠ 0. É o passo que o Great Expectations deixa manual — escrever a suíte de expectativas é o trabalho difícil, e é onde a ferramenta morre na vida real.

```bash
python -m sdd emit     data/contas.parquet --target inadimplente --out contrato.yaml
python -m sdd validate data/safra_nova.parquet --contract contrato.yaml
```

71 checks de 5 tipos saem de 26 colunas: `proporcao` (26), `faixa` (19), `psi` (15), `niveis` (6), `sentinela` (5). Nenhum carrega severidade atribuída pelo emissor — ela vem do `impact`, sempre etiquetada com o nível de evidência.

Verificação em três cenários (`bench/contract_check.py`):

| cenário | falhas | esperado |
|---|---:|---|
| a própria base que gerou o contrato | **0** | 0 |
| amostra limpa de 10 000 linhas | **0** | 0 |
| safra deteriorada com 6 mudanças conhecidas | 9 | 6/6 pegas + 3 consequências rastreáveis |

As 3 falhas extras no cenário 3 não são ruído: `score_bureau.faixa` e `.distribuicao` são consequência de a mesma transformação ter preenchido os nulos com valores fora da faixa histórica, e `taxa_juros.faixa` do mesmo deslocamento que disparou o PSI.

### Tolerância não pode ser fixa no contrato

O desenho ingênuo — e o que quase toda suíte faz — é guardar o intervalo calculado na emissão e exigir que o lote novo caia dentro dele. Com 5 M linhas, o intervalo de Wilson sobre a taxa de nulo tem meia-largura de **±0,03 p.p.**, tão apertado que ruído amostral puro o viola:

| linhas do lote (limpo, amostrado da própria referência) | tolerância fixa | duas proporções |
|---:|---:|---:|
| 1 000 | 2 alarmes | **0** |
| 10 000 | 2 alarmes | **0** |
| 100 000 | 1 alarme | **0** |
| 1 000 000 | 1 alarme | **0** |

Eu esperava que o problema fosse específico de lote pequeno. Não é — **falha em todo tamanho**, inclusive com 1 M de linhas, porque o intervalo da emissão é apertado demais para qualquer reamostragem. E os alarmes caem sempre nas duas únicas colunas que têm nulos, ou seja: sob tolerância fixa, *toda* coluna com nulo real alarma sem que nada tenha mudado.

A correção é o contrato guardar `p_ref` e `n_ref` e o check ser um **teste de duas proporções** avaliado com o *n* do lote novo — a incerteza dos dois lados entra na conta. Mais duas defesas contra o ruído que faz gente desligar alerta: **α corrigida por Bonferroni** sobre os 71 checks (1,4 × 10⁻⁴ por check), e **piso de efeito prático** — com milhões de linhas a significância sozinha alarma por 0,02 p.p. que não muda decisão nenhuma, então todo teste exige significância *e* efeito mínimo.

---

## Impacto na decisão: a tese, em uma tabela

Toda ferramenta de qualidade de dados ordena por prevalência. Se ordenar por impacto desse a mesma lista, o projeto inteiro não se justificaria. Mesma base, mesmos achados, duas ordenações:

```
 #  POR PREVALÊNCIA (o que toda ferramenta faz)   #  POR IMPACTO NA DECISÃO           move
------------------------------------------------------------------------------------------
 1. valor_recuperado = 0.0                 72,2%  1. score_bureau            0,03878    +1
 2. score_bureau                           38,7%  2. valor_recuperado = 0.0  0,03039    -1
 3. valor_fatura                            9,1%  3. utilizacao = -1.0       0,01619    +4
 4. score = 0.001                           4,9%  4. valor_fatura            0,00909    -1
 5. renda = -999.0                          3,1%  5. renda = -999.0          0,00312     =
 6. uf = 'N/A'                              2,0%  6. uf = 'N/A'              0,00198     =
 7. utilizacao = -1.0                       2,0%  7. data_nascimento         0,00150    +1
 8. data_nascimento = '1900-01-01'          1,5%  8. idade = 0.0             0,00122    +1
 9. idade = 0.0                             1,2%  9. dias_desde_ultimo_pgto  0,00080    +1
10. dias_desde_ultimo_pgto = 9999.0         0,8% 10. score = 0.001           0,00000    -6
```

Os dois extremos são o argumento inteiro:

- **`utilizacao = -1`** afeta 1,98% das linhas — sétimo lugar por prevalência, atrás de defeitos 4x maiores. Mas está **8,16x mais concentrado na margem da decisão** que a taxa-base, e sobe para terceiro. É um defeito injetado de propósito para isso: 25% de chance quando o escore está em [0,62; 0,78], 0,4% fora.
- **`score = 0.001`** afeta 4,88% das linhas — quarto por prevalência — e inverte **exatamente zero decisões**. O corte está em 0,70; os valores clipados em 0,001 estão longe demais para atravessar, e nenhum reparo os move. Cai para último.

`python bench/impact_check.py`

### A escada de evidência

Para afirmar "corrigir a coluna X muda a decisão" é preciso a função de X para decisão — e essa função é o modelo. Não existe atalho que dê o número exato sem ela. O que existe é uma escada, e **cada achado carrega a etiqueta do degrau que produziu seu número**:

| nível | o que mede | quando vale | unidade |
|---|---|---|---|
| **A** | `massa × alavancagem univariada` | sempre, com alvo | ordinal — só ordena, **não** é percentual de decisão |
| **B1** | taxa de inversão exata | defeito na própria coluna que decide | fração de linhas que trocam de lado |
| **B2** | `P(defeito E na margem)` | defeito em outra coluna | **limite superior** do que poderia inverter |
| C | taxa de inversão com o modelo | marco 5 | fração real |

A alavancagem do nível A é `2·|AUC − 0,5|`, com AUC univariada exata (Mann-Whitney por tabela de contingência, que trata empate corretamente) — 2,3 s para 16 colunas em 5 M linhas, barato o bastante para não aproximar. Em coluna categórica a AUC sai de ordenar os níveis pela taxa de evento observada, o que é otimista em amostra, e fica marcada como tal.

O nível B2 define "margem" em **percentil do escore** (banda de ±5 p.p. em torno do corte), então a taxa-base é 10% por construção e o lift sai direto — `lift = 1,00` significa que o defeito está espalhado de forma indiferente à decisão, e a exposição é exatamente massa/10. Foi assim que se descobriu que **todos os defeitos originais da base tinham lift ≈ 1**: a métrica não estava errada, era a base que não tinha nenhum caso interessante. Sem o defeito injetado na margem, não haveria evidência de que o nível B discrimina coisa alguma.

Efeito colateral do nível A: `valor_recuperado` e `n_acionamentos_cobranca` saem com **AUC exatamente 1,0000**. É vazamento — a coluna responde o alvo sozinha. O detector próprio é o D4, marco 5; aqui o sinal só cai de graça do cálculo da alavancagem, e o relatório diz isso em vez de reivindicar a detecção.

---

## Seis achados que contrariaram o desenho inicial

### 1. "Número de passadas" é a métrica de custo errada

O instinto é empacotar tudo numa query só para varrer o dado uma vez. Medido:

| | 1 query | em lotes |
|---|---:|---:|
| 400 `count(*) FILTER` | 16,66 s | **1,81 s** (lotes de 16) |
| 24 `approx_top_k` | 11,68 s | **1,14 s** (lotes de 2) |
| 104 agregados simples | **1,13 s** | 2,44 s (lotes de 16) |

As duas famílias se comportam de forma **oposta**. Agregado barato é limitado por I/O e divide a mesma varredura — empacote ao máximo. Agregado com estado (Space-Saving, t-digest, cadeia de `FILTER`) degrada de forma super-linear na mesma query — lote pequeno, mesmo custando mais varreduras.

O motor passou a agrupar expressões **por classe de custo**, com lote calibrado por classe. Resultado: 28,0 s → 4,1 s no mesmo perfil, com saída idêntica. Ressalva honesta: com a base cabendo em cache de página, revarrer é quase de graça e o custo de expressão domina. Em base maior que a RAM o balanço muda — daí o `--batch-profile io`.

`python bench/batching.py`

### 2. Cardinalidade aproximada não serve para relatório

`approx_count_distinct` (HyperLogLog) do DuckDB, contra o exato:

| coluna | HLL | exato | erro |
|---|---:|---:|---:|
| `valor_fatura` | 640 514 | 773 260 | **17,2%** |
| `cep` | 4 212 145 | 4 876 000 | 13,6% |
| `n_atrasos_12m` | 10 | 9 | 11,1% |

Longe dos ~0,8% que a literatura de HLL promete, e o DuckDB não expõe parâmetro de precisão. Só que `count(DISTINCT)` **exato custa 1,3 s** para as 26 colunas nesta escala — mais barato do que o erro justifica.

O motor passou a ser adaptativo: coluna estreita ganha cardinalidade exata de graça (o `histogram()` já enumerou os valores), coluna média ganha `count(DISTINCT)` exato, e o HLL fica só para a coluna larga demais para materializar, marcada com `~` no relatório. Custo total: +1,1 s. Nesta base, 24 de 26 colunas saem exatas.

`python bench/accuracy.py`

### 3. O quantil aproximado quebra exatamente onde o detector precisa dele

`approx_quantile(dias_desde_ultimo_pgto, 0.99)`, mesma query, mesma base, várias execuções:

```
threads = 1        89, 89, 89, 89, 89          (89 = valor exato)
threads = 4        89, 89, 89, 206, 89
threads = todas    672, 178, 1503, 89, 89
```

Duas coisas ao mesmo tempo. O t-digest é mesclado em paralelo e a ordem da mesclagem muda o resultado — **não é determinístico**. E o erro se concentra no quantil vizinho a uma massa pontual: 0,8% da coluna vale 9999, e o algoritmo interpola entre centroides devolvendo valores que não existem no dado (682, 1096, 1503 — os valores reais são ≤ 89 ou 9999).

Fatal para o teste de gap do detector D1, que compara o candidato a sentinela contra P01/P99 e o IQR. Um detector cujo veredito muda entre execuções não presta.

A correção reforçou a arquitetura: a camada 2 tinha sido desenhada para **conter custo**, e é também onde se **compra acurácia de volta**. O gap test usa `quantile_cont` exato, restrito às poucas colunas com candidato — 0,16 s para uma coluna, 0,79 s para três, contra 13 s para a base inteira. O perfil geral mantém o t-digest, marcado como aproximado.

Detalhe adicional: `approx_quantile(x, [p1..pn])` erra mais que `approx_quantile(x, p)` chamado n vezes (682 contra 89 no mesmo dado), mas a forma escalar custa 11x mais. Fica a lista, com a etiqueta.

### 4. O plano de várias queries só se paga onde revarrer é barato

O achado 1 troca varreduras por lotes menores, e isso pressupõe que revarrer é quase de graça. Em Parquet é: colunar, com projection pushdown e cache de página. Em CSV não — **cada query reparseia o arquivo inteiro**. O mesmo perfil, em 200 mil linhas de CSV, custava 43,6 s contra 5,0 s de 5 milhões de linhas de Parquet: 200x pior por linha.

Uma varredura do CSV custa 0,71 s; materializar em tabela temporária custa 0,79 s **uma vez** e derruba as varreduras seguintes para ~0 s. Com isso: **43,6 s → 1,33 s**, sem tocar em nada do resto do motor.

A regra que ficou: materializar por padrão quando a fonte não é colunar, nunca quando é (`--materialize auto|sim|nao`). O corolário desconfortável é que o achado 1 não é universal — ele é uma propriedade do par *motor × formato de origem*, não do motor sozinho.

### 5. A escala do teste de gap decide o veredito — e a escolha óbvia o inverte

O DESIGN especificava o teste de gap como `|v − v'| > 3·IQR`. Medido, o IQR **inverte o veredito**:

| coluna | v | gap/IQR | gap/amplitude | lift | verdade |
|---|---:|---:|---:|---:|---|
| `renda` | −999 | 0,59 | 0,011 | 33.657 | sentinela |
| `idade` | 0 | 0,46 | 0,234 | 1 | sentinela |
| `dias_desde_ultimo_pgto` | 9999 | 220,2 | 111,3 | 1 | sentinela |
| `qtd_produtos` | 0 | **1,00** | 0,091 | 3 | **legítimo** |
| `valor_recuperado` | 0 | 0,00 | 0,000 | 251.597 | zero estrutural |

A armadilha `qtd_produtos = 0` pontua **1,00 contra 0,46** da sentinela `idade = 0`: com IQR, o valor legítimo é considerado *mais* isolado que o defeito real. E `gap/amplitude` também não resolve sozinha — `renda` (0,011) fica abaixo da armadilha (0,091).

O que separa é usar **escalas diferentes conforme a coluna seja discreta ou contínua**, o que coincide exatamente com a divisão de rotas que a camada 1 já produzia por motivo de custo:

- **discreta** (histograma completo) → escala = **espaçamento mediano entre valores adjacentes**. Mede buraco no suporte, que é o que "ponto isolado" significa. `idade` 18x e `dias` 9.910x contra a armadilha em 1x.
- **contínua** (top-k) → escala = **amplitude do suporte restante**, com o *lift* (massa × ndv) como condição necessária. Separa `renda` de `valor_recuperado` por cinco ordens de grandeza.

Dois corolários que ficaram no código. Primeiro: **massa não pode ser critério de candidatura em coluna discreta** — `idade = 0` tem 1,2% contra 1,27% dos valores legítimos, então ranquear por massa perderia a sentinela; é preciso testar todo valor, o que só é viável porque o histograma já está na memória. Segundo: **massa pontual sem isolamento não é sentinela** — é distribuição mista (`valor_recuperado` é 72% zeros porque só inadimplente tem recuperação). Sai como achado próprio, porque quebra premissa de modelo do mesmo jeito, mas o veredito de sentinela exige o gap. Distinguir os dois casos exige semântica que a ferramenta não tem, e ela não finge ter.

### 6. O contrato reprovava a base que o gerou

O PSI precisa de uma distribuição esperada. A minha vinha dos **níveis de probabilidade dos quantis** — se as bordas são p05, p25, p50…, a massa entre p25 e p50 é 25% por construção. Correto, e só vale se as bordas forem todas distintas.

Em coluna discreta elas não são: `n_atrasos_12m` tem p001 = p01 = p05 = p25 = p50 = 0. Colapsar bordas repetidas é obrigatório, mas depois do colapso os níveis de probabilidade deixam de descrever os bins. Medido na própria base de referência:

```
esperado inferido  [0,0010, 0,7490, 0,2000, 0,0400, 0,0090, 0,0010]
proporção real     [0,0000, 0,5488, 0,3290, 0,0989, 0,0199, 0,0034]
```

PSI da referência contra ela mesma: **≈ 0,13**, acima do limite de 0,10. **O contrato reprovava a base que o produziu**, sem nenhum lote novo envolvido — e o alarme apareceria em toda coluna discreta, que é a maioria delas. É exatamente assim que uma suíte de validação vira ruído e é desligada na segunda semana.

A correção é medir em vez de inferir: uma query em lote na emissão apura a proporção real de cada bin na referência. E dela saiu uma segunda: **`identificador` não tem distribuição que valha checar**. O PSI de `id_conta` deu 10,2 só porque o lote novo tem outras chaves — tecnicamente certo, informativamente nulo. Colunas com ndv/n > 0,5 não recebem check de distribuição.

O que fechou o buraco de verdade, porém, foi a invariante que o bug sugeriu: **`emit` agora valida o contrato contra a própria referência antes de gravá-lo**, e se recusa a gravar se algum check reprovar. Parece trivial; foi o que transformou um erro silencioso em falha imediata.

---

## Uso

```bash
python -m sdd profile  data/contas.parquet --metadata-only   # camada 0, milissegundos
python -m sdd profile  data/contas.parquet --json perfil.json
python -m sdd detect   data/contas.parquet --target inadimplente
python -m sdd emit     data/contas.parquet --out contrato.yaml
python -m sdd validate data/safra_nova.parquet --contract contrato.yaml
python -m sdd impact   data/contas.parquet --target inadimplente                        --decision-column score --decision-cutoff 0.70
```

```
[D1] renda = -999.0
     ponto isolado: vizinho mais proximo 845.44, a 1.11% da amplitude do suporte
     (166,730); tratar como ausente afeta 3.11% das linhas
     confianca 0.97 | sentinela_numerica
[D3] score_bureau
     taxa de nulo salta de 2.0% para 100.0% em 2025-04-01; nao e uma coluna com
     39% de nulos, e uma quebra de origem
     confianca 0.97 | nulo_por_tempo
```

Todo achado carrega **a evidência que o produziu**, não um rótulo. E nenhum carrega severidade vinda do detector: ela sai do `impact`, com a etiqueta do nível de evidência que a produziu.

```
fonte     data/contas.parquet
linhas    5,000,000   colunas 26
camada 0  31,998 B de footer lidos de 207.4 MB (0.0154%) em 3 ms | 10 row groups
camada 1  22 queries | 5.15 s | duckdb 1.5.5
          por classe: distinct 1.59s  filter 0.99s  quantile 0.83s  cheap 0.71s  topk 0.59s

CONCENTRACAO DE MASSA (valor mais frequente por coluna)
  coluna                  valor          massa  x uniforme      contagem  cobertura
  renda                   -999.0         3.11%     33,657x       155,491  top-k
  data_nascimento         1900-01-01     1.50%        439x        75,191  top-k
```

`x uniforme` é a massa dividida por 1/ndv. Alto significa **moda**, não sentinela — o veredito exige o teste de gap, que é o marco 2. A ferramenta não finge saber mais do que sabe.

## Dados

`bench/make_data.py` gera uma base sintética de gestão de conta com **12 defeitos injetados e um manifesto com o gabarito** — sentinelas numéricas com e sem gap, sentinela textual e temporal, quebra de pipeline por data, nulo correlacionado com categoria, duas colunas de vazamento e uma armadilha deliberada (`qtd_produtos = 0`, zero legítimo em coluna de contagem, que um detector baseado em lista de constantes canônicas vai marcar como falso positivo).

O manifesto é o gabarito do benchmark de precisão e recall do marco 2 — a parte que separa isto de uma ferramenta e faz virar projeto de ciência de dados. A demonstração em dado real usa o Lending Club, onde os vazamentos já são documentados na literatura e o detector D4 tem que reencontrá-los sozinho.

```bash
python bench/make_data.py --rows 5_000_000 --out data/contas.parquet
```

## Instalação

```bash
python -m venv .venv && .venv/Scripts/pip install -e .
```

Dependências: `duckdb`, `pyarrow`. Para os benchmarks, `numpy` e `psutil`; a comparação com o ydata-profiling exige um venv separado em Python 3.12 (ele não instala em 3.14).

## Não-objetivos

- Relatório HTML com 200 métricas por coluna. Já existe e não é o valor.
- Suporte a "qualquer fonte". Um formato colunar bem feito vale mais que cinco conectores meia-boca.
- Correção automática do dado. A ferramenta detecta, quantifica e contrata; quem repara é o dono do pipeline.
- Substituir o Great Expectations. Ele executa contratos — este projeto **os escreve**.
