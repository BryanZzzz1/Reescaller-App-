@echo off
REM ---------------------------------------------------------------------------
REM  Genera el ejecutable distribuible de Reescaller.
REM
REM    crear_exe.bat              -> compila y crea dist\Reescaller + ZIP (+ instalador si hay Inno Setup)
REM    crear_exe.bat -SkipBuild   -> solo vuelve a montar dist\ con lo ya compilado
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0."
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\build_dist.ps1" %*
echo.
pause
endlocal
