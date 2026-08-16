"""Estatistica do contrato — de onde sai a tolerancia.

Uma suite de validacao com tolerancia chutada (`±10%`) e desligada em duas semanas:
ou nao pega nada, ou alarma toda madrugada. A tolerancia tem que sair do dado.

Tres decisoes que este modulo carrega:

**1. Tolerancia nao pode ser fixa no contrato.** O intervalo de Wilson sobre a taxa de
nulo com n = 5 M e +-0,03 p.p.; com n = 10 mil e +-0,6 p.p. Um contrato emitido de uma
base grande e validado contra um lote pequeno alarmaria sem parar se guardasse o
intervalo da emissao. Entao o contrato guarda **p_ref e n_ref**, e o teste e de **duas
proporcoes**, avaliado com o n do lote novo. O intervalo de Wilson aparece no YAML como
documentacao do que foi observado, nao como criterio.

**2. Significancia estatistica nao e significancia pratica.** Com 5 M linhas, uma
diferenca de 0,02 p.p. e altissimamente significativa e nao muda decisao nenhuma. Sem
um piso de efeito, o contrato vira gerador de ruido justamente nas bases grandes, onde
ele mais deveria servir. Por isso todo teste exige **significancia E efeito minimo**.
O piso e declarado; a versao principiada — piso derivado do impacto na decisao — e o
marco 4, e ate la o default e uma constante honesta e documentada como tal.

**3. Multiplicidade.** Um contrato com 26 colunas gera ~100 checks. A alpha = 0,05,
seriam ~5 alarmes falsos por execucao — o suficiente para alguem desligar tudo. A alpha
e corrigida por Bonferroni sobre o numero de checks do contrato.
"""

from __future__ import annotations

from math import erfc, log, sqrt

Z_95 = 1.959963984540054


def wilson(p: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Intervalo de Wilson para uma proporcao. Vai no YAML como documentacao."""
    if n <= 0:
        return (0.0, 1.0)
    d = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meio = z / d * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centro - meio), min(1.0, centro + meio))


def normal_sf(x: float) -> float:
    """Cauda superior da normal padrao, via erfc — sem depender de scipy."""
    return 0.5 * erfc(x / sqrt(2.0))


def duas_proporcoes(p_ref: float, n_ref: int, p_novo: float, n_novo: int) -> tuple[float, float]:
    """Teste z de duas proporcoes. Devolve (z, p-valor bilateral).

    A hipotese nula e que os dois lotes vem do mesmo processo. Diferente de comparar o
    lote novo contra um intervalo fixo: aqui a incerteza dos DOIS lados entra na conta,
    que e o que impede o alarme espurio quando o lote novo e pequeno.
    """
    if n_ref <= 0 or n_novo <= 0:
        return (0.0, 1.0)
    p = (p_ref * n_ref + p_novo * n_novo) / (n_ref + n_novo)
    se = sqrt(p * (1 - p) * (1 / n_ref + 1 / n_novo))
    if se <= 0:
        return (0.0, 1.0 if p_ref == p_novo else 0.0)
    z = (p_novo - p_ref) / se
    return (z, 2 * normal_sf(abs(z)))


def psi(esperado: list[float], observado: list[float], eps: float = 1e-6) -> float:
    """Population Stability Index sobre proporcoes ja binadas.

    Os bins vem dos quantis do lote de referencia, entao `esperado` e conhecido por
    construcao e nao precisa ser estimado. Faixas usuais em risco: < 0,10 estavel,
    0,10-0,25 atencao, > 0,25 mudanca material.

    Ressalva herdada do credit-risk-decision-engine: PSI agregado sobre o escore e
    pouco sensivel — la, sob safra deteriorada, a aprovacao caiu 7 p.p. com PSI em
    0,027. Por isso aqui ele e sempre **por feature**, nunca so sobre a saida.
    """
    total = 0.0
    for e, o in zip(esperado, observado):
        e = max(e, eps)
        o = max(o, eps)
        total += (o - e) * log(o / e)
    return total


def alpha_corrigida(alpha: float, n_checks: int) -> float:
    """Bonferroni. Conservador de proposito: o custo de um alarme falso aqui nao e
    estatistico, e alguem desligar o contrato."""
    return alpha / max(1, n_checks)
