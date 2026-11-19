"""Genera licenses/RUST_CRATES.txt: licencia y textos de cada dependencia de Rust del workspace.

Uso:  python -I scripts\\gen_third_party.py

Lee `cargo metadata` (necesita `cargo` en el PATH o en ~/.cargo/bin y las dependencias ya descargadas con `cargo fetch`/compilar)
y, por cada crate externo, escribe nombre, versión, expresión SPDX, repositorio y el contenido de sus archivos
LICENSE / COPYING / NOTICE. Es lo que exigen MIT, Apache-2.0, BSD, etc. al redistribuir los binarios.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "licenses" / "RUST_CRATES.txt"
PATRON = re.compile(r"^(LICEN[CS]E|COPYING|UNLICENSE|NOTICE)([-_.].*)?$", re.I)


def cargo():
    exe = shutil.which("cargo")
    if exe:
        return exe
    for c in (Path.home() / ".cargo" / "bin" / "cargo.exe", Path.home() / ".cargo" / "bin" / "cargo"):
        if c.exists():
            return str(c)
    sys.exit("No se encontró cargo.")


def main():
    # En Windows con la toolchain GNU, cargo metadata no necesita MinGW; solo lee Cargo.lock y el registro local.
    r = subprocess.run(
        [cargo(), "metadata", "--format-version", "1", "--filter-platform", "x86_64-pc-windows-gnu"],
        cwd=ROOT / "rust", capture_output=True, text=True, encoding="utf-8",
    )
    if r.returncode != 0:
        sys.exit(r.stderr)
    meta = json.loads(r.stdout)
    propios = set(meta["workspace_members"])
    paquetes = sorted((p for p in meta["packages"] if p["id"] not in propios), key=lambda p: (p["name"], p["version"]))

    lineas = [
        "DEPENDENCIAS DE RUST DE REESCALLER",
        "===================================",
        "",
        f"{len(paquetes)} crates de terceros. Generado con scripts/gen_third_party.py a partir de rust/Cargo.lock.",
        "Cada crate se distribuye bajo la licencia (o una de las licencias) indicada abajo; se incluye el texto que trae.",
        "",
        "RESUMEN",
        "-------",
    ]
    for p in paquetes:
        lineas.append(f"{p['name']} {p['version']}  --  {p.get('license') or '(ver archivos de licencia)'}")
    lineas += ["", ""]

    sin_texto = []
    for p in paquetes:
        lineas += ["=" * 100, f"{p['name']} {p['version']}", f"Licencia: {p.get('license') or '(no declarada)'}"]
        if p.get("repository"):
            lineas.append(f"Repositorio: {p['repository']}")
        lineas.append("=" * 100)
        base = Path(p["manifest_path"]).parent
        archivos = sorted(f for f in base.iterdir() if f.is_file() and PATRON.match(f.name)) if base.exists() else []
        if p.get("license_file"):
            lf = base / p["license_file"]
            if lf.is_file() and lf not in archivos:
                archivos.append(lf)
        if not archivos:
            sin_texto.append(f"{p['name']} {p['version']}")
            lineas += ["(El paquete no incluye un archivo de licencia; se aplica la licencia SPDX indicada arriba.)", ""]
            continue
        for f in archivos:
            lineas += [f"--- {f.name} ---", f.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").strip(), ""]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"{len(paquetes)} crates -> {OUT} ({OUT.stat().st_size // 1024} KB)")
    if sin_texto:
        print(f"{len(sin_texto)} sin archivo de licencia propio (solo SPDX):", ", ".join(sin_texto[:12]))


if __name__ == "__main__":
    main()
