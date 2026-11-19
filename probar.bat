@echo off
REM ---------------------------------------------------------------------------
REM Verificador automatico del port a Rust.
REM
REM   probar.bat          -> comprueba entorno + reescalado + peso + lote + video
REM   probar.bat --keep   -> conserva los archivos generados para inspeccionarlos
REM ---------------------------------------------------------------------------
setlocal
REM Ojo: %~dp0 termina en "\" y rompe el escapado de comillas de Windows,
REM por eso se usa "%~dp0." (misma carpeta, sin barra final).
cd /d "%~dp0."

if not exist "rust\target\release\reescaller-selftest.exe" (
    echo [1/2] Compilando el verificador ^(solo la primera vez^)...
    call "rust\build.bat" build --release
    if errorlevel 1 (
        echo ERROR: la compilacion fallo.
        pause
        exit /b 1
    )
)

echo [2/2] Ejecutando las comprobaciones...
echo.
REM Ojo: %~dp0 termina en "\" y rompe el escapado de comillas de Windows,
REM por eso se pasa "%~dp0." (misma carpeta, sin barra final).
rust\target\release\reescaller-selftest.exe --project "%~dp0." %*
echo.
pause
endlocal
