@echo off
REM ---------------------------------------------------------------------------
REM Compilador helper para Windows + Rust (toolchain GNU).
REM
REM Motivo: desde Git Bash (MSYS) las rutas se corrompen al pasarlas a los
REM procesos nativos, y rustc no encontraba "dlltool.exe". Este script prepara
REM el PATH en formato Windows nativo antes de invocar a cargo.
REM ---------------------------------------------------------------------------
set "MINGW=C:\Users\BBRYA\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
set "PATH=%MINGW%;%USERPROFILE%\.cargo\bin;%PATH%"
cd /d "%~dp0"
cargo.exe %*
