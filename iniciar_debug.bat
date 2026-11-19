@echo off
REM ---------------------------------------------------------------------------
REM  Interfaz de depuración (Rust) - lanzador
REM
REM    iniciar_debug.bat            -> build release + servidor en primer plano
REM    iniciar_debug.bat --debug    -> build de depuración (compila rápido)
REM    iniciar_debug.bat --port 8080-> otro puerto
REM
REM  OJO: %~dp0 termina en "\" y rompe el escapado de comillas de Windows,
REM  por eso se usa "%~dp0." (misma carpeta, sin barra final).
REM ---------------------------------------------------------------------------

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0."
title Reescaller - Interfaz de depuracion (Rust)

set "PORT=7878"
set "BUILD=release"
set "BIN=rust\target\release\reescaller-debug.exe"
set "RUST_BUILD=rust\build.bat"

:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--debug" (
    set "BUILD=debug"
    set "BIN=rust\target\debug\reescaller-debug.exe"
    shift
    goto parse
)
if /i "%~1"=="--port" (
    set "PORT=%~2"
    shift
    shift
    goto parse
)
if /i "%~1"=="--help" goto usage
shift
goto parse
:parsed

echo ============================================================
echo  Reescaller ^(Rust^) - Interfaz de depuracion
echo ============================================================
echo  Puerto : %PORT%
echo  Build  : %BUILD%
echo  Proyecto: %CD%
echo ============================================================
echo.

REM ---------------------------------------------------------------------------
REM 1) Cerrar cualquier instancia anterior (el puerto 7878 se queda libre)
REM ---------------------------------------------------------------------------
taskkill /F /IM reescaller-debug.exe >nul 2>&1
if %errorlevel%==0 (
    echo [i] Se cerro una instancia anterior de reescaller-debug.exe.
    timeout /t 1 /nobreak >nul
)

REM ---------------------------------------------------------------------------
REM 2) Compilar si falta el binario
REM ---------------------------------------------------------------------------
if not exist "%BIN%" (
    echo [1/3] Compilando la interfaz de depuracion. Puede tardar varios minutos...
    call "%RUST_BUILD%" build --%BUILD%
    if errorlevel 1 (
        echo.
        echo [X] La compilacion fallo. Revisa el log anterior.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Binario encontrado: %BIN%
)

REM ---------------------------------------------------------------------------
REM 3) Comprobar que el puerto está libre
REM ---------------------------------------------------------------------------
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do set "PID_USE=%%p"
if defined PID_USE (
    echo.
    echo [X] El puerto %PORT% ya esta en uso por el proceso !PID_USE! ^(PID^).
    echo     Cierra ese programa o usa: iniciar_debug.bat --port 8080
    echo.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM 4) Abrir el navegador una vez el servidor esté listo
REM ---------------------------------------------------------------------------
echo [2/3] Iniciando el servidor...
start "" /B cmd /c "timeout /t 3 /nobreak >nul & start "" http://127.0.0.1:%PORT%"

echo [3/3] Interfaz de depuracion: http://127.0.0.1:%PORT%
echo.
echo     · Para detener el servidor: pulsa Ctrl+C aqui.
echo     · Para cerrarlo de otra forma: taskkill /F /IM reescaller-debug.exe
echo     · En otra terminal puedes lanzar:  rust\target\%BUILD%\reescaller.exe doctor
echo.

REM ---------------------------------------------------------------------------
REM 5) Servidor en primer plano (los logs se ven en esta ventana)
REM ---------------------------------------------------------------------------
"%BIN%" --port %PORT% --project "%~dp0."
set "CODIGO=%errorlevel%"

echo.
if "%CODIGO%"=="0" (
    echo El servidor de depuracion se ha cerrado correctamente.
) else (
    echo [X] El servidor termino con el codigo %CODIGO%.
    echo     Si dice "puerto en uso", hay otra instancia corriendo:
    echo     taskkill /F /IM reescaller-debug.exe
)
echo.
pause
endlocal
exit /b %CODIGO%

:usage
echo Uso: iniciar_debug.bat [--debug] [--port 8080]
endlocal
