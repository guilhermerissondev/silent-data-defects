"""Gera uma base sintetica de gestao de conta com defeitos injetados conhecidos.

Escreve um Parquet particionado em row groups e um manifesto JSON com a verdade
de campo: cada defeito injetado, a coluna, a massa e o detector que deveria pega-lo.
O manifesto e o gabarito do benchmark de precisao/recall (marco 2).

    python bench/make_data.py --rows 5_000_000 --out data/contas.parquet
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EPOCH = date(1970, 1, 1)
UFS = np.array(["SP", "MG", "RJ", "BA", "PR", "RS", "PE", "CE", "SC", "GO", "PA", "MA"])
UF_P = np.array([0.22, 0.11, 0.10, 0.08, 0.07, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.11])
PRODUTOS = np.array(["rotativo", "parcelado", "consignado", "cheque_especial"])
CANAIS = np.array(["app", "agencia", "web", "telefone", "parceiro"])

# Gabarito do benchmark. `esperado` diz se o detector DEVE disparar: a entrada com
# esperado=False e armadilha, e disparar nela conta como falso positivo.
DEFECTS = [
    dict(id="renda_sentinela", coluna="renda", detector="D1", tipo="sentinela_numerica",
         valor=-999.0, massa=0.031, esperado=True,
         nota="gap: rendas reais comecam em ~800"),
    dict(id="idade_zero", coluna="idade", detector="D1", tipo="sentinela_numerica",
         valor=0, massa=0.012, esperado=True,
         nota="gap: idades reais em [18, 95]"),
    dict(id="dias_pgto_9999", coluna="dias_desde_ultimo_pgto", detector="D1",
         tipo="sentinela_numerica", valor=9999, massa=0.008, esperado=True,
         nota="sentinela no extremo superior"),
    dict(id="qtd_produtos_zero", coluna="qtd_produtos", detector="D1",
         tipo="sentinela_numerica", valor=0, massa=0.26, esperado=False,
         nota="ARMADILHA: zero legitimo em coluna de contagem, sem gap, com massa ALTA. "
              "Detector que olha constante canonica ou massa vai marcar; so o "
              "espacamento do suporte rejeita."),
    dict(id="utilizacao_sentinela_na_margem", coluna="utilizacao", detector="D1",
         tipo="sentinela_numerica", valor=-1.0, massa=0.017, esperado=True,
         nota="massa PEQUENA mas concentrada na margem da decisao: 25% de chance quando "
              "o escore esta em [0,62; 0,78], 0,4% fora. Existe para testar o nivel B — "
              "deve ficar acima de defeitos muito mais massivos na ordenacao por impacto."),
    dict(id="uf_na", coluna="uf", detector="D2", tipo="sentinela_textual",
         valor="N/A", massa=0.020, esperado=True, nota=""),
    dict(id="nascimento_1900", coluna="data_nascimento", detector="D2",
         tipo="sentinela_temporal", valor="1900-01-01", massa=0.015, esperado=True, nota=""),
    # Massa pontual legitima: o gerador clipa `score` em [0.001, 0.999] e so
    # inadimplente tem recuperacao. Nao sao sentinelas, mas sao distribuicoes mistas
    # reais e o detector deve reporta-las como tal — nao como sentinela.
    dict(id="score_clip", coluna="score", detector="D1",
         tipo="massa_pontual_em_continua", valor=None, massa=0.049, esperado=True,
         nota="clipping do escore cria massa pontual nas duas pontas"),
    dict(id="recuperado_zero", coluna="valor_recuperado", detector="D1",
         tipo="massa_pontual_em_continua", valor=0.0, massa=0.72, esperado=True,
         nota="inflacao de zero estrutural: so inadimplente tem recuperacao"),
    dict(id="bureau_quebra", coluna="score_bureau", detector="D3",
         tipo="nulo_por_tempo", valor=None, massa=None, esperado=True,
         nota="nulo salta de ~2% para 100% a partir de 2025-04-01"),
    dict(id="fatura_nulo_produto", coluna="valor_fatura", detector="D3",
         tipo="nulo_por_categoria", valor=None, massa=None, esperado=True,
         nota="nulo concentrado em produto=consignado"),
    dict(id="recuperado_vaza", coluna="valor_recuperado", detector="D4",
         tipo="vazamento", valor=None, massa=None, esperado=True,
         nota="so e > 0 quando inadimplente=1; AUC univariada ~1.0"),
    dict(id="acionamentos_vaza", coluna="n_acionamentos_cobranca", detector="D4",
         tipo="vazamento", valor=None, massa=None, esperado=True,
         nota="posterior ao desfecho"),
    dict(id="cep_cardinalidade", coluna="cep", detector="D6",
         tipo="cardinalidade_alta", valor=None, massa=None, esperado=True, nota=""),
    dict(id="versao_constante", coluna="versao_modelo", detector="D0",
         tipo="coluna_constante", valor="v3.1", massa=1.0, esperado=True,
         nota="detectavel na camada 0, sem ler linha"),
]

# Colunas sem nenhum defeito. Qualquer achado aqui e falso positivo.
CLEAN_COLUMNS = [
    "id_conta", "data_ref", "saldo_atual", "limite", "n_atrasos_12m",
    "tempo_relacionamento_meses", "produto", "canal", "possui_seguro", "flag_pep",
    "taxa_juros", "inadimplente",
]


def _days(d: date) -> int:
    return (d - EPOCH).days


def make_chunk(rng: np.random.Generator, n: int, offset: int) -> dict[str, pa.Array]:
    # --- nucleo limpo -------------------------------------------------------
    id_conta = np.arange(offset, offset + n, dtype=np.int64)

    # 24 safras mensais entre 2024-01 e 2025-12
    safra = rng.integers(0, 24, n)
    data_ref = np.array([_days(date(2024 + m // 12, m % 12 + 1, 1)) for m in range(24)])[safra]

    idade = rng.integers(18, 96, n).astype(np.int32)
    renda = np.round(np.exp(rng.normal(8.0, 0.75, n)) + 800.0, 2)
    tempo_rel = rng.integers(1, 300, n).astype(np.int32)
    n_atrasos = rng.poisson(0.6, n).astype(np.int32)
    limite = np.round(renda * rng.uniform(0.4, 3.5, n), 2)
    utilizacao = np.clip(rng.beta(2.0, 3.0, n), 0, 1)
    saldo_atual = np.round(limite * utilizacao, 2)
    qtd_produtos = rng.poisson(1.4, n).astype(np.int32)
    dias_pgto = rng.integers(0, 90, n).astype(np.int32)
    taxa_juros = np.round(rng.normal(0.12, 0.03, n), 4)
    valor_fatura = np.round(saldo_atual * rng.uniform(0.05, 1.0, n), 2)

    uf = rng.choice(UFS, n, p=UF_P)
    produto = rng.choice(PRODUTOS, n, p=[0.45, 0.30, 0.15, 0.10])
    canal = rng.choice(CANAIS, n, p=[0.40, 0.15, 0.25, 0.10, 0.10])
    cep = np.char.zfill((rng.integers(1_000_000, 99_999_999, n)).astype(str), 8)
    possui_seguro = rng.random(n) < 0.23
    flag_pep = rng.random(n) < 0.004
    versao_modelo = np.full(n, "v3.1")

    data_nascimento = data_ref - (idade.astype(np.int64) * 365 + rng.integers(0, 365, n))

    # score_bureau correlacionado com risco latente
    latente = (
        -0.35 * np.log(renda)
        + 1.10 * utilizacao
        + 0.45 * n_atrasos
        - 0.004 * tempo_rel
        + rng.normal(0, 0.8, n)
    )
    score_bureau = np.clip(
        750 - 60 * (latente - latente.mean()) / (latente.std() + 1e-9), 300, 1000
    ).astype(np.int32)

    z = (latente - latente.mean()) / (latente.std() + 1e-9)
    # intercepto calibrado para ~22% de eventos, mesma ordem do credit-risk-decision-engine
    pd_true = 1.0 / (1.0 + np.exp(-(1.3 * z - 1.25)))
    inadimplente = (rng.random(n) < pd_true).astype(np.int8)
    # escore do modelo: PD verdadeira com ruido, e a coluna que a regra de decisao usa
    score = np.clip(pd_true + rng.normal(0, 0.06, n), 0.001, 0.999).round(5)

    # --- defeitos injetados -------------------------------------------------
    # D4 vazamento: so existe depois do desfecho
    inad = inadimplente == 1
    valor_recuperado = np.zeros(n)
    valor_recuperado[inad] = np.round(saldo_atual[inad] * rng.uniform(0, 0.6, inad.sum()), 2)
    n_acionamentos = np.zeros(n, dtype=np.int32)
    n_acionamentos[inad] = rng.integers(1, 15, inad.sum()).astype(np.int32)

    # D1 renda = -999 (gap real: rendas legitimas >= 800)
    renda = renda.copy()
    renda[rng.random(n) < 0.031] = -999.0

    # D1 utilizacao = -1, concentrada na MARGEM da decisao (corte do escore em 0,70).
    # Massa total pequena, mas quase toda em cima da fronteira: e o caso que separa
    # impacto de prevalencia, e o unico defeito desta base com lift > 1 no nivel B.
    na_margem = (score >= 0.62) & (score <= 0.78)
    p_util = np.where(na_margem, 0.25, 0.004)
    utilizacao = np.where(rng.random(n) < p_util, -1.0, np.round(utilizacao, 4))

    # D1 idade = 0 (gap real: idades legitimas >= 18)
    idade[rng.random(n) < 0.012] = 0

    # D1 dias_desde_ultimo_pgto = 9999 (extremo superior)
    dias_pgto[rng.random(n) < 0.008] = 9999

    # ARMADILHA: qtd_produtos = 0 sem gap. Poisson ja produz zeros legitimos;
    # aqui so reforcamos a massa. Nao e sentinela.
    qtd_produtos[rng.random(n) < 0.02] = 0

    # D2 uf = "N/A"
    uf = np.where(rng.random(n) < 0.020, "N/A", uf)

    # D2 data_nascimento = 1900-01-01
    data_nascimento = np.where(rng.random(n) < 0.015, _days(date(1900, 1, 1)), data_nascimento)

    # D3 score_bureau: quebra de pipeline a partir de 2025-04-01
    quebra = data_ref >= _days(date(2025, 4, 1))
    bureau_nulo = quebra | (rng.random(n) < 0.02)

    # D3 valor_fatura: nulo concentrado em consignado
    p_nulo_fatura = np.where(produto == "consignado", 0.55, 0.01)
    fatura_nulo = rng.random(n) < p_nulo_fatura

    return {
        "id_conta": pa.array(id_conta),
        "data_ref": pa.array(data_ref.astype(np.int32), type=pa.date32()),
        "idade": pa.array(idade),
        "data_nascimento": pa.array(data_nascimento.astype(np.int32), type=pa.date32()),
        "renda": pa.array(renda),
        "score_bureau": pa.array(np.where(bureau_nulo, None, score_bureau), type=pa.int32()),
        "saldo_atual": pa.array(saldo_atual),
        "limite": pa.array(limite),
        "utilizacao": pa.array(utilizacao),
        "valor_fatura": pa.array(np.where(fatura_nulo, None, valor_fatura), type=pa.float64()),
        "n_atrasos_12m": pa.array(n_atrasos),
        "tempo_relacionamento_meses": pa.array(tempo_rel),
        "dias_desde_ultimo_pgto": pa.array(dias_pgto),
        "qtd_produtos": pa.array(qtd_produtos),
        "taxa_juros": pa.array(taxa_juros),
        "uf": pa.array(uf),
        "produto": pa.array(produto),
        "canal": pa.array(canal),
        "cep": pa.array(cep),
        "possui_seguro": pa.array(possui_seguro),
        "flag_pep": pa.array(flag_pep),
        "versao_modelo": pa.array(versao_modelo),
        "valor_recuperado": pa.array(valor_recuperado),
        "n_acionamentos_cobranca": pa.array(n_acionamentos),
        "score": pa.array(score),
        "inadimplente": pa.array(inadimplente),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--out", type=Path, default=Path("data/contas.parquet"))
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    writer = None
    written = 0
    try:
        while written < args.rows:
            n = min(args.chunk, args.rows - written)
            batch = pa.table(make_chunk(rng, n, written))
            if writer is None:
                writer = pq.ParquetWriter(args.out, batch.schema, compression="zstd")
            writer.write_table(batch)
            written += n
            print(f"  {written:>12,} / {args.rows:,}", end="\r")
    finally:
        if writer is not None:
            writer.close()

    manifest = {
        "rows": args.rows,
        "seed": args.seed,
        "path": str(args.out),
        "defects": DEFECTS,
        "clean_columns": CLEAN_COLUMNS,
        "decision": {"kind": "threshold", "column": "score", "cutoff": 0.70},
        "target": "inadimplente",
        "time_column": "data_ref",
    }
    mpath = args.out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    size_mb = args.out.stat().st_size / 1e6
    print(f"\n{args.out}  {size_mb:,.1f} MB  {args.rows:,} linhas")
    print(f"{mpath}  {len(DEFECTS)} defeitos no gabarito")


if __name__ == "__main__":
    main()
