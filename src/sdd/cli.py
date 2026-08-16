"""CLI do sdd. Marco 1: `profile`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .scan import duckdb_pass, metadata
from .types import ScanResult


def _fmt(v, width: int = 14) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        s = f"{v:,.4g}"
    elif isinstance(v, int):
        s = f"{v:,}"
    else:
        s = str(v)
    return s if len(s) <= width else s[: width - 1] + "~"


def render(res: ScanResult, meta_findings: list[dict], show_mass: float) -> str:
    out: list[str] = []
    md = res.metadata
    n_cols = len(res.columns) or (len(md.columns) if md else 0)
    out.append(f"fonte     {res.source}")
    out.append(f"linhas    {res.rows:,}   colunas {n_cols}")
    if md is not None:
        pct = 100 * md.footer_bytes / md.file_bytes if md.file_bytes else 0
        out.append(
            f"camada 0  {md.footer_bytes:,} B de footer lidos de {md.file_bytes / 1e6:,.1f} MB "
            f"({pct:.4f}%) em {md.elapsed_s * 1000:,.0f} ms | {md.row_groups} row groups"
        )
    if res.queries:
        det = "  ".join(f"{k} {v:.2f}s" for k, v in sorted(res.timing_by_class.items(),
                                                           key=lambda kv: -kv[1]))
        out.append(f"camada 1  {res.queries} queries | {res.elapsed_s:,.2f} s | {res.engine}")
        if det:
            out.append(f"          por classe de expressao: {det}")
    out.append("")

    if meta_findings:
        out.append("ACHADOS DA CAMADA 0 (custo zero, sem ler linha)")
        for f in meta_findings:
            out.append(f"  [{f['achado']:<22}] {f['coluna']:<26} {f['detalhe']}")
        out.append("")

    hdr = f"{'coluna':<28}{'tipo':<12}{'nulo%':>8}{'ndv':>12}{'min':>16}{'max':>16}"
    out.append(hdr)
    out.append("-" * len(hdr))
    if res.columns:
        for name, cs in res.columns.items():
            # '~' marca cardinalidade estimada por HyperLogLog; sem marca, e exata
            ndv = _fmt(cs.ndv_approx, 11) + ("" if cs.ndv_exact else "~")
            out.append(
                f"{name:<28}{cs.kind:<12}{cs.null_rate * 100:>7.2f}%"
                f"{ndv:>12}{_fmt(cs.min, 16):>16}{_fmt(cs.max, 16):>16}"
            )
        aprox = [c.name for c in res.columns.values() if not c.ndv_exact and c.ndv_approx]
        if aprox:
            out.append(f"  ~ = ndv estimado por HyperLogLog (erro medido ate 17%): "
                       f"{', '.join(aprox)}")
        if any(c.quantiles for c in res.columns.values()):
            out.append("  quantis vem de t-digest: aproximados e nao deterministicos com "
                       "threads > 1.")
            out.append("  Perto de massa pontual o erro explode (p99 = 89 exato, 89 a 1.503 "
                       "aproximado). Ver sdd/scan/exact.py.")
    elif md is not None:
        # camada 0 sozinha: nulo% e min/max sao exatos, ndv so se o writer gravou
        for name, cm in md.columns.items():
            nr = cm.null_rate
            nrs = f"{nr * 100:>7.2f}%" if nr is not None else f"{'-':>8}"
            out.append(
                f"{name:<28}{cm.physical_type.lower():<12}{nrs}"
                f"{_fmt(cm.distinct_count, 12):>12}{_fmt(cm.min, 16):>16}{_fmt(cm.max, 16):>16}"
            )

    # Concentracao de massa: insumo bruto do detector D1 (marco 2), ainda sem veredito.
    # Uma linha por coluna — o valor mais frequente e quanto ele pesa. Listar todos os
    # valores acima de um limiar afoga a saida em modas legitimas de coluna discreta:
    # e justamente por isso que o veredito depende do teste de gap, nao da massa.
    conc = []
    for name, cs in res.columns.items():
        if not cs.value_counts or not cs.non_null:
            continue
        v, c = max(cs.value_counts.items(), key=lambda kv: kv[1])
        m = c / cs.non_null
        if m >= show_mass:
            esperado = 1 / cs.ndv_approx if cs.ndv_approx else None
            conc.append((m, name, v, c, esperado, cs.counts_source))
    if conc:
        conc.sort(reverse=True)
        out.append("")
        out.append(f"CONCENTRACAO DE MASSA (valor mais frequente por coluna, massa >= {show_mass:.1%})")
        out.append(f"  {'coluna':<28}{'valor':<20}{'massa':>9}{'x uniforme':>12}"
                   f"{'contagem':>14}  cobertura")
        for m, name, v, c, esp, src_ in conc:
            razao = f"{m / esp:>11,.0f}x" if esp else f"{'-':>12}"
            cob = "completa" if src_ == "histogram" else "top-k"
            out.append(f"  {name:<28}{_fmt(v, 20):<20}{m:>8.2%}{razao}{c:>14,}  {cob}")
        out.append("  'x uniforme' = massa observada dividida por 1/ndv. Alto so significa moda,")
        out.append("  nao sentinela — o veredito exige o teste de gap (D1, marco 2).")
    return "\n".join(out)


def cmd_profile(args: argparse.Namespace) -> int:
    paths = metadata.resolve(args.path)
    if not paths:
        print(f"nada encontrado em {args.path}", file=sys.stderr)
        return 2

    meta_scan = None
    findings: list[dict] = []
    if metadata.is_parquet(paths):
        meta_scan = metadata.scan_metadata(paths)
        findings = metadata.metadata_findings(meta_scan)
    elif args.metadata_only:
        print("camada 0 exige Parquet; CSV nao tem footer com estatistica", file=sys.stderr)
        return 2

    if args.metadata_only:
        res = ScanResult(source=paths[0] if len(paths) == 1 else f"{len(paths)} arquivos",
                         rows=meta_scan.rows, metadata=meta_scan, engine="pyarrow (footer)")
        print(render(res, findings, args.min_mass))
        return 0

    mat = {"auto": None, "sim": True, "nao": False}[args.materialize]
    res = duckdb_pass.scan(paths, threads=args.threads, memory_limit=args.memory_limit,
                           value_counts=not args.no_value_counts,
                           batch_profile=args.batch_profile,
                           exact_ndv=not args.approx_ndv, materialize=mat)
    res.metadata = meta_scan
    print(render(res, findings, args.min_mass))

    if args.json:
        payload = res.to_dict()
        payload["metadata_findings"] = findings
        Path(args.json).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nperfil salvo em {args.json}")
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    from .detect import missingness, sentinels
    from .detect.base import Finding

    paths = metadata.resolve(args.path)
    if not paths:
        print(f"nada encontrado em {args.path}", file=sys.stderr)
        return 2

    achados: list[Finding] = []
    if metadata.is_parquet(paths):
        for f in metadata.metadata_findings(metadata.scan_metadata(paths)):
            if f["achado"] in ("coluna_constante", "coluna_100pct_nula", "schema_divergente"):
                achados.append(Finding(coluna=f["coluna"], detector="D0", tipo=f["achado"],
                                       confianca=f["confianca"],
                                       evidencia={"camada": 0, "custo": "zero linhas lidas"},
                                       nota=f["detalhe"]))

    res = duckdb_pass.scan(paths, threads=args.threads, memory_limit=args.memory_limit)
    con = duckdb_pass.connect(args.threads, args.memory_limit)
    src = duckdb_pass.source_expr(paths)
    achados += sentinels.detect(con, src, res.columns, min_mass=args.min_mass)
    achados += missingness.detect(con, src, res.columns, time_col=args.time_column,
                                  target=args.target)

    print(f"{res.rows:,} linhas x {len(res.columns)} colunas | perfil {res.elapsed_s:.2f}s")
    tc = args.time_column or missingness.pick_time_column(res.columns)
    if tc:
        print(f"coluna de tempo: {tc}" + ("" if args.time_column else " (inferida)"))
    if not args.target:
        print("sem --target: o teste de nulo informativo nao roda")
    print()

    if not achados:
        print("nenhum achado acima dos limiares")
        return 0

    ordem = {"D0": 0, "D1": 1, "D2": 2, "D3": 3}
    for f in sorted(achados, key=lambda f: (ordem.get(f.detector, 9), -f.confianca)):
        print(f"[{f.detector}] {f.coluna}" + ("" if f.valor is None else f" = {f.valor!r}"))
        print(f"     {f.nota}")
        if args.verbose:
            print(f"     confianca {f.confianca:.2f} | {f.tipo} | {f.evidencia}")
        else:
            print(f"     confianca {f.confianca:.2f} | {f.tipo}")

    print(f"\n{len(achados)} achados. Severidade nao aparece aqui: ela deriva do impacto")
    print("na decisao, que e o marco 4. Ordenar por prevalencia seria o erro que este")
    print("projeto existe para nao cometer.")

    if args.json:
        Path(args.json).write_text(json.dumps([f.to_dict() for f in achados],
                                              indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nachados salvos em {args.json}")
    return 0


def _perfil_e_achados(paths, args):
    from .detect import missingness, sentinels
    from .detect.base import Finding

    achados: list = []
    if metadata.is_parquet(paths):
        for f in metadata.metadata_findings(metadata.scan_metadata(paths)):
            if f["achado"] in ("coluna_constante", "coluna_100pct_nula", "schema_divergente"):
                achados.append(Finding(coluna=f["coluna"], detector="D0", tipo=f["achado"],
                                       confianca=f["confianca"], evidencia={"camada": 0},
                                       nota=f["detalhe"]))
    res = duckdb_pass.scan(paths, threads=args.threads, memory_limit=args.memory_limit)
    con = duckdb_pass.connect(args.threads, args.memory_limit)
    src = duckdb_pass.source_expr(paths)
    achados += sentinels.detect(con, src, res.columns)
    achados += missingness.detect(con, src, res.columns,
                                  time_col=getattr(args, "time_column", None),
                                  target=getattr(args, "target", None))
    return res, achados, con, src


def cmd_impact(args: argparse.Namespace) -> int:
    from . import impact

    paths = metadata.resolve(args.path)
    if not paths:
        print(f"nada encontrado em {args.path}", file=sys.stderr)
        return 2

    res, achados, con, src = _perfil_e_achados(paths, args)
    decisao = None
    if args.decision_column:
        decisao = {"tipo": args.decision_kind, "coluna": args.decision_column}
        if args.decision_kind == "corte":
            if args.decision_cutoff is None:
                print("--decision-kind corte exige --decision-cutoff", file=sys.stderr)
                return 2
            decisao["corte"] = args.decision_cutoff
        else:
            if args.decision_fraction is None:
                print("--decision-kind top_k exige --decision-fraction", file=sys.stderr)
                return 2
            decisao["fracao"] = args.decision_fraction

    linhas = impact.avaliar(con, src, achados, res.columns, alvo=args.target,
                            decisao=decisao)

    print(f"{res.rows:,} linhas x {len(res.columns)} colunas")
    if decisao:
        fr = next((l["nivel_b"]["fronteira"] for l in linhas if l["nivel_b"]), None)
        d = f"{decisao['coluna']} >= {decisao.get('corte')}" if decisao["tipo"] == "corte" \
            else f"top {decisao.get('fracao'):.1%} de {decisao['coluna']}"
        print(f"decisao   {d}" + (f" | corte no percentil {fr['percentil_do_corte']:.1%}, "
                                  f"margem em +-{100*(fr['banda_percentil'][1]-fr['percentil_do_corte']):.0f} p.p."
                                  if fr else ""))
    else:
        print("sem --decision-column: so o nivel A (ordenacao), nunca impacto medido")
    if not args.target:
        print("sem --target: sem alavancagem, o nivel A nao roda")
    print()

    hdr = (f"{'':<2}{'achado':<42}{'nivel':>6}{'impacto':>10}{'massa':>9}"
           f"{'lift':>8}  significado")
    print(hdr)
    print("-" * 112)
    for l in linhas:
        f, imp = l["achado"], l["impacto"]
        rot = f"[{f.detector}] {f.coluna}" + ("" if f.valor is None else f" = {f.valor!r}")
        if imp is None:
            print(f"  {rot:<42}{'-':>6}{'-':>10}{'-':>9}{'-':>8}  sem evidencia de impacto")
            continue
        sub = imp.get("sub", "A")
        rotulo = {"B1_inversao_exata": "inversao exata",
                  "B2_exposicao": "exposicao (limite superior)"}.get(sub, "ordinal")
        massa = imp.get("massa_afetada", imp.get("massa"))
        lift = imp.get("lift_na_margem")
        ls = f"{lift:>7.2f}x" if lift is not None else f"{'-':>8}"
        print(f"  {rot:<42}{imp['nivel']:>6}{imp['valor']:>10.5f}{massa:>8.2%}{ls}  {rotulo}")

    print("\nA e B nao medem a mesma coisa e nao tem a mesma unidade.")
    print("  A = massa x alavancagem univariada. Ordena a lista. NAO e percentual de decisao.")
    print("  B1 = taxa de inversao exata (defeito na propria coluna que decide).")
    print("  B2 = P(defeito E na margem). LIMITE SUPERIOR do que poderia inverter.")
    print("  Propagar defeito de outra coluna ate a decisao exige o modelo: nivel C, marco 5.")

    vaza = [n for n, v in (l["alavancagem"] or {}).items() if False] if False else \
        sorted({l["achado"].coluna for l in linhas
                if (l["alavancagem"] or {}).get("suspeita_de_vazamento")})
    if vaza:
        print(f"\nAUC univariada >= 0.95 em: {', '.join(vaza)} — a coluna praticamente")
        print("responde o alvo sozinha. O detector proprio e o D4 (marco 5); aqui o sinal")
        print("so cai de graca do calculo da alavancagem.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"achado": l["achado"].to_dict(), "impacto": l["impacto"],
              "nivel_a": l["nivel_a"], "nivel_b": l["nivel_b"]} for l in linhas],
            indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nsalvo em {args.json}")
    return 0


def cmd_emit(args: argparse.Namespace) -> int:
    from .contract import emit

    paths = metadata.resolve(args.path)
    if not paths:
        print(f"nada encontrado em {args.path}", file=sys.stderr)
        return 2

    res, achados, con, src = _perfil_e_achados(paths, args)
    decisao = None
    if args.decision_column:
        decisao = {"tipo": "corte", "coluna": args.decision_column, "corte": args.decision_cutoff}
    contrato = emit.emitir(res, achados, fonte=str(args.path), con=con, src=src,
                           alpha=args.alpha, efeito_taxa=args.min_effect, decisao=decisao)

    n_ck = len(contrato["checks"])
    tipos: dict[str, int] = {}
    for c in contrato["checks"]:
        tipos[c["tipo"]] = tipos.get(c["tipo"], 0) + 1
    print(f"{res.rows:,} linhas x {len(res.columns)} colunas | perfil {res.elapsed_s:.2f}s")
    print(f"{n_ck} checks emitidos: " + ", ".join(f"{k} {v}" for k, v in sorted(tipos.items())))
    print(f"alpha {args.alpha} corrigida por Bonferroni -> {args.alpha / n_ck:.2e} por check")
    print(f"efeito minimo {args.min_effect:.1%} (constante declarada; deriva-la do "
          f"impacto na decisao e o marco 4)")

    # Auto-teste: o contrato tem que passar na base que o gerou. Parece trivial e nao e —
    # foi assim que se descobriu que as proporcoes esperadas do PSI estavam inferidas em
    # vez de medidas, e que o contrato reprovava a propria referencia sem nenhum lote
    # novo envolvido. Um contrato que nao passa aqui esta quebrado por construcao.
    if not args.no_self_check:
        from .contract import validate as val
        falhas = [r for r in val.validar(con, src, contrato) if not r.passou]
        if falhas:
            print(f"\nAUTO-TESTE FALHOU: {len(falhas)} checks reprovam a propria base de "
                  f"referencia. O contrato esta errado, nao o dado.")
            for r in falhas:
                print(f"  [{r.tipo}] {r.id}: {r.detalhe}")
            print("\ncontrato NAO gravado")
            return 1
        print(f"auto-teste ok: os {n_ck} checks passam na base que os gerou")

    Path(args.out).write_text(_dump(contrato, args.out), encoding="utf-8")
    print(f"\ncontrato em {args.out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from .contract import validate as val

    paths = metadata.resolve(args.path)
    if not paths:
        print(f"nada encontrado em {args.path}", file=sys.stderr)
        return 2
    contrato = _load(args.contract)

    con = duckdb_pass.connect(args.threads, args.memory_limit)
    src = duckdb_pass.source_expr(paths)
    if not metadata.is_parquet(paths):
        con.execute(f"CREATE TEMP TABLE _sdd_src AS SELECT * FROM {src}")
        src = "_sdd_src"
    n = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]

    res = val.validar(con, src, contrato)
    falhas = [r for r in res if not r.passou]

    ref = contrato["fonte"]
    print(f"contrato  {args.contract} | emitido de {ref['linhas']:,} linhas")
    print(f"lote      {args.path} | {n:,} linhas")
    print(f"politica  alpha {contrato['politica']['alpha']} / {len(contrato['checks'])} "
          f"checks (Bonferroni), efeito minimo "
          f"{contrato['politica']['efeito_minimo']['taxa']:.1%}\n")

    if falhas:
        print(f"FALHOU  {len(falhas)} de {len(res)} checks\n")
        for r in falhas:
            print(f"  [{r.tipo:<10}] {r.id}")
            print(f"               {r.detalhe}")
            if args.verbose:
                print(f"               {r.evidencia}")
    else:
        print(f"OK  {len(res)} checks passaram")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"id": r.id, "coluna": r.coluna, "tipo": r.tipo, "passou": r.passou,
              "observado": r.observado, "referencia": r.referencia,
              "detalhe": r.detalhe, "evidencia": r.evidencia} for r in res],
            indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\nresultado em {args.json}")
    return 1 if falhas else 0


def _dump(obj: dict, destino: str) -> str:
    if str(destino).endswith((".yaml", ".yml")):
        try:
            import yaml
            return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, width=100)
        except ImportError:
            print("pyyaml nao instalado; gravando JSON no arquivo indicado",
                  file=sys.stderr)
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _load(caminho: str) -> dict:
    texto = Path(caminho).read_text(encoding="utf-8")
    if caminho.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(texto)
    return json.loads(texto)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sdd", description="silent-data-defects")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="perfila uma base (camada 0 + camada 1)")
    p.add_argument("path", help="arquivo, glob ou diretorio")
    p.add_argument("--metadata-only", action="store_true",
                   help="so a camada 0: footer do Parquet, zero linhas lidas")
    p.add_argument("--no-value-counts", action="store_true",
                   help="pula quantis e contagem de valores (so os agregados baratos)")
    p.add_argument("--batch-profile", choices=("cpu", "io"), default="cpu",
                   help="cpu: base cabe em cache, lotes pequenos nas classes pesadas. "
                        "io: base maior que a RAM, empacota mais para revarrer menos")
    p.add_argument("--materialize", choices=("auto", "sim", "nao"), default="auto",
                   help="carrega a fonte em tabela temporaria antes de perfilar. "
                        "auto = sim para CSV (revarrer reparseia), nao para Parquet")
    p.add_argument("--approx-ndv", action="store_true",
                   help="usa so HyperLogLog para cardinalidade (erro medido ate 17%%); "
                        "por padrao o exato e usado onde a tabela hash cabe")
    p.add_argument("--min-mass", type=float, default=0.005,
                   help="massa minima para listar um valor concentrado (default 0.5%%)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--memory-limit", default=None, help="ex: 2GB")
    p.add_argument("--json", default=None, help="salva o perfil completo em JSON")
    p.set_defaults(func=cmd_profile)

    d = sub.add_parser("detect", help="roda os detectores sobre a base")
    d.add_argument("path", help="arquivo, glob ou diretorio")
    d.add_argument("--target", default=None,
                   help="coluna de alvo binario, para o teste de nulo informativo")
    d.add_argument("--time-column", default=None,
                   help="coluna de data de referencia; inferida se omitida")
    d.add_argument("--min-mass", type=float, default=0.001,
                   help="massa minima de um valor para virar candidato (default 0.1%%)")
    d.add_argument("--verbose", action="store_true", help="mostra a evidencia completa")
    d.add_argument("--threads", type=int, default=None)
    d.add_argument("--memory-limit", default=None)
    d.add_argument("--json", default=None, help="salva os achados em JSON")
    d.set_defaults(func=cmd_detect)

    e = sub.add_parser("emit", help="emite o contrato de validacao a partir do perfil")
    e.add_argument("path", help="arquivo, glob ou diretorio de referencia")
    e.add_argument("--out", default="contract.yaml", help="destino (.yaml ou .json)")
    e.add_argument("--target", default=None)
    e.add_argument("--time-column", default=None)
    e.add_argument("--alpha", type=float, default=0.01,
                   help="nivel do teste antes da correcao de multiplicidade")
    e.add_argument("--min-effect", type=float, default=0.005,
                   help="efeito minimo em proporcao para um check falhar (default 0.5%%)")
    e.add_argument("--decision-column", default=None,
                   help="coluna de escore da regra de decisao, registrada no contrato")
    e.add_argument("--decision-cutoff", type=float, default=None)
    e.add_argument("--no-self-check", action="store_true",
                   help="pula a verificacao de que o contrato passa na propria referencia")
    e.add_argument("--threads", type=int, default=None)
    e.add_argument("--memory-limit", default=None)
    e.set_defaults(func=cmd_emit)

    v = sub.add_parser("validate", help="roda o contrato contra um lote novo")
    v.add_argument("path", help="lote a validar")
    v.add_argument("--contract", required=True, help="contrato emitido")
    v.add_argument("--verbose", action="store_true", help="mostra a evidencia de cada falha")
    v.add_argument("--json", default=None, help="salva o resultado completo")
    v.add_argument("--threads", type=int, default=None)
    v.add_argument("--memory-limit", default=None)
    v.set_defaults(func=cmd_validate)

    i = sub.add_parser("impact", help="ordena os achados por impacto na decisao")
    i.add_argument("path", help="arquivo, glob ou diretorio")
    i.add_argument("--target", default=None, help="alvo binario, para a alavancagem (nivel A)")
    i.add_argument("--time-column", default=None)
    i.add_argument("--decision-column", default=None,
                   help="coluna de escore que a regra de decisao usa (nivel B)")
    i.add_argument("--decision-kind", choices=("corte", "top_k"), default="corte")
    i.add_argument("--decision-cutoff", type=float, default=None)
    i.add_argument("--decision-fraction", type=float, default=None,
                   help="fracao revisada, para --decision-kind top_k")
    i.add_argument("--json", default=None)
    i.add_argument("--threads", type=int, default=None)
    i.add_argument("--memory-limit", default=None)
    i.set_defaults(func=cmd_impact)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
