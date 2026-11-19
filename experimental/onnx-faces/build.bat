@echo off
REM Compila la prueba de concepto con el mismo toolchain GNU de la app.
set "MINGW=C:\Users\BBRYA\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
set "PATH=%MINGW%;%USERPROFILE%\.cargo\bin;%PATH%"
cd /d "%~dp0rust"
cargo.exe %*
