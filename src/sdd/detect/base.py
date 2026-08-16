"""Vocabulario comum dos detectores.

Um achado descreve **o que foi observado e com que evidencia**, nunca o quanto importa.
Severidade nao aparece aqui de proposito: ela deriva do impacto na decisao (marco 4),
e um detector que se auto-atribui severidade e um detector que ordena por prevalencia —
exatamente o que este projeto existe para nao fazer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from ..types import _plain


@dataclass
class Finding:
    coluna: str
    detector: str            # D0..D6
    tipo: str                # sentinela_numerica, nulo_por_tempo, ...
    valor: Any = None        # o valor implicado, quando ha um
    massa: float | None = None
    confianca: float = 0.0
    evidencia: dict[str, Any] = field(default_factory=dict)
    nota: str = ""

    def to_dict(self) -> dict:
        return _plain(asdict(self))

    def __str__(self) -> str:
        v = "" if self.valor is None else f" = {self.valor!r}"
        m = "" if self.massa is None else f" ({self.massa:.2%})"
        return f"[{self.detector}] {self.coluna}{v}{m}  conf {self.confianca:.2f}  {self.nota}"


# Constantes que costumam codificar ausencia. NAO sao criterio de deteccao —
# entram so como reforco de confianca depois que o teste de gap ja decidiu.
# Usar esta lista como criterio seria regex disfarcada de detector, e e justamente
# o que faz um detector marcar `qtd_produtos = 0` como sentinela.
CANONICAL = {-1, -9, -99, -999, -9999, -99999, -999999, -9999999, -99999999,
             -999999999, 9999, 99999, 999999, 9999999, 99999999, 999999999,
             -2147483648, 2147483647, -32768, 32767}


def is_canonical(v: Any) -> bool:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return f.is_integer() and int(f) in CANONICAL
