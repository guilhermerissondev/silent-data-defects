"""Camada 2 — sob demanda, so nas colunas que a camada 1 marcou.

A camada 2 nasceu para conter custo: agregado caro roda em 5-15 colunas de 200, nao
em todas. A medicao mostrou que ela tambem e onde se compra acuracia de volta.

`approx_quantile` do DuckDB, sobre 5 M linhas com 0,8% de massa no valor 9999:

    threads = 1      p99 em 5 execucoes: 89, 89, 89, 89, 89          (89 = exato)
    threads = 4      p99 em 5 execucoes: 89, 89, 89, 206, 89
    threads = todas  p99 em 5 execucoes: 672, 178, 1503, 89, 89

O t-digest e mesclado em paralelo e a ordem da mesclagem muda o resultado; o erro se
concentra justamente no quantil vizinho a uma massa pontual, porque o algoritmo
interpola entre centroides e devolve um valor que nao existe no dado (682, 1096, 1503
sao todos inventados: os valores reais sao <= 89 ou 9999).

Isso e fatal para o teste de gap do D1, que compara o candidato a sentinela contra
P01/P99 e o IQR: um detector cujo veredito muda entre execucoes nao presta. Entao o
gap test nao usa quantil aproximado — usa `quantile_cont` exato, restrito as poucas
colunas com candidato. Custa 0,16 s para uma coluna e 0,79 s para tres, contra 13 s
para a base inteira.

Nota de forma de chamada: `approx_quantile(x, [p1, ..., pn])` erra mais que
`approx_quantile(x, p)` chamado n vezes — 682 contra 89 no mesmo dado. A forma
escalar, porem, custa 11x mais (9,2 s contra 0,83 s para 17 colunas), o que nao
compensa: para o perfil geral fica a lista, marcada como aproximada, e quem precisa
de numero confiavel vem para ca.
"""

from __future__ import annotations

from typing import Any, Sequence

from .duckdb_pass import BATCH_CPU, q

# Poucas colunas por query: quantile_cont materializa a coluna ordenada e empilhar
# varios na mesma query degrada do mesmo jeito que as outras classes pesadas.
EXACT_BATCH = 3


def exact_quantiles(con, src: str, columns: Sequence[str],
                    probs: Sequence[float] = (0.01, 0.25, 0.5, 0.75, 0.99),
                    ) -> dict[str, dict[float, Any]]:
    """Quantis exatos e deterministicos, para as colunas que chegaram na camada 2."""
    out: dict[str, dict[float, Any]] = {}
    ps = ", ".join(str(p) for p in probs)
    cols = list(columns)
    for s in range(0, len(cols), EXACT_BATCH):
        chunk = cols[s:s + EXACT_BATCH]
        sel = ", ".join(f"quantile_cont({q(c)}, [{ps}]) AS x{i}"
                        for i, c in enumerate(chunk))
        row = con.execute(f"SELECT {sel} FROM {src}").fetchone()
        for c, vals in zip(chunk, row):
            out[c] = dict(zip(probs, vals or []))
    return out


def nearest_neighbours(con, src: str, column: str, value: Any) -> tuple[Any, Any]:
    """Vizinhos imediatos de `value` no suporte da coluna — insumo do teste de gap.

    Sentinela e um ponto isolado: renda com 0 e depois nada ate 800 e codigo de
    ausente; renda com 0, 12, 40, 95 e valor legitimo. E a distancia ate o vizinho,
    nao o valor em si, que separa os dois casos.
    """
    c = q(column)
    row = con.execute(
        f"SELECT max({c}) FILTER (WHERE {c} < {value!r}), "
        f"       min({c}) FILTER (WHERE {c} > {value!r}) FROM {src}"
    ).fetchone()
    return row[0], row[1]
