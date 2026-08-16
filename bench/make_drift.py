"""Gera uma safra deteriorada a partir da base, com mudancas conhecidas.

Serve para testar o `validate`: cada transformacao aqui deve aparecer como um check
falho, e nada alem delas deve falhar. O lote sai **menor** que a base de propositio —
e o caso que quebra contrato de tolerancia fixa, porque o ruido amostral do lote novo
passa a dominar.

    python bench/make_drift.py --rows 1_000_000 --out data/safra_nova.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

MUDANCAS = [
    dict(coluna="renda", tipo="sentinela_cresce", check="renda.sentinela[-999.0]",
         detalhe="massa de -999 sobe de ~3,1% para ~8%: a origem piorou"),
    dict(coluna="uf", tipo="nivel_novo", check="uf.niveis",
         detalhe="nivel 'EX' aparece em ~0,5% das linhas"),
    dict(coluna="taxa_juros", tipo="deslocamento", check="taxa_juros.distribuicao",
         detalhe="distribuicao deslocada em +0,03: PSI deve disparar"),
    dict(coluna="idade", tipo="fora_da_faixa", check="idade.faixa",
         detalhe="~0,2% das linhas com idade 120, acima do maximo observado"),
    dict(coluna="limite", tipo="coluna_removida", check="limite.presente",
         detalhe="coluna sumiu do lote"),
    dict(coluna="score_bureau", tipo="nulo_corrigido", check="score_bureau.nulo",
         detalhe="quebra de pipeline resolvida: nulo volta de ~39% para ~2%. "
                 "Mudanca real e para melhor — o contrato deve apontar, nao ignorar"),
]

SQL = """
SELECT
  id_conta, data_ref,
  CASE WHEN hash(id_conta * 7) % 1000 < 2 THEN 120 ELSE idade END        AS idade,
  data_nascimento,
  CASE WHEN hash(id_conta * 3) % 1000 < 80 THEN -999.0 ELSE renda END    AS renda,
  coalesce(score_bureau,
           CASE WHEN hash(id_conta * 11) % 1000 < 20 THEN NULL
                ELSE 300 + (hash(id_conta * 13) % 700)::INTEGER END)     AS score_bureau,
  saldo_atual,
  utilizacao, valor_fatura, n_atrasos_12m, tempo_relacionamento_meses,
  dias_desde_ultimo_pgto, qtd_produtos,
  taxa_juros + 0.03                                                      AS taxa_juros,
  CASE WHEN hash(id_conta * 17) % 1000 < 5 THEN 'EX' ELSE uf END         AS uf,
  produto, canal, cep, possui_seguro, flag_pep, versao_modelo,
  valor_recuperado, n_acionamentos_cobranca, score, inadimplente
FROM read_parquet(?)
LIMIT ?
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/contas.parquet")
    ap.add_argument("--out", type=Path, default=Path("data/safra_nova.parquet"))
    ap.add_argument("--rows", type=int, default=1_000_000)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"COPY ({SQL.replace('?', '{}').format(repr(args.base), args.rows)}) "
                f"TO '{args.out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    manifesto = {"base": args.base, "rows": args.rows, "path": str(args.out),
                 "mudancas": MUDANCAS}
    mp = args.out.with_suffix(".mudancas.json")
    mp.write_text(json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{args.out}  {args.out.stat().st_size / 1e6:,.1f} MB  {args.rows:,} linhas")
    print(f"{mp}  {len(MUDANCAS)} mudancas conhecidas:")
    for m in MUDANCAS:
        print(f"  {m['check']:<32}{m['detalhe']}")


if __name__ == "__main__":
    main()
