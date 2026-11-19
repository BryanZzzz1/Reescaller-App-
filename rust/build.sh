#!/bin/bash
# Helper de compilación para Windows + Rust (toolchain GNU).
#
# Motivo: en Git Bash (MSYS) las rutas con barras mixtas se corrompen al
# pasarlas a procesos nativos, por lo que rustc no encontraba `dlltool.exe`.
# Aquí forzamos PATH en formato Windows nativo con MSYS2_ENV_CONV_EXCL="*".
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

MINGW_WIN='C:\Users\BBRYA\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin'
CARGO_WIN='C:\Users\BBRYA\.cargo\bin'

WIN_BASE=$(cygpath -w "$PATH")
export MSYS2_ENV_CONV_EXCL="*"
export PATH="$CARGO_WIN;$MINGW_WIN;$WIN_BASE"

cd "$SCRIPT_DIR"
exec /c/Users/BBRYA/.cargo/bin/cargo.exe "$@"
