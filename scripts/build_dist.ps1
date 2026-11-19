<#
.SYNOPSIS
  Genera la carpeta distribuible de Reescaller (portable) y, opcionalmente, el instalador.

.DESCRIPTION
  1. Compila en release (toolchain GNU + MinGW, igual que rust\build.bat).
  2. Monta dist\Reescaller\ con Reescaller.exe, el motor de IA, FFmpeg y los modelos.
  3. Copia vcomp140.dll junto al motor (el motor la necesita y un PC limpio no la trae).
  4. Crea dist\Reescaller-<version>-win64.zip.
  5. Si encuentra Inno Setup (ISCC.exe) compila también installer\Reescaller.iss.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\build_dist.ps1
  powershell -ExecutionPolicy Bypass -File scripts\build_dist.ps1 -SkipBuild
#>
param(
    [switch]$SkipBuild,
    [string]$TargetDir = "",
    [string]$DistRoot = "",
    [switch]$WithOnnx,   # incluye el motor de rostros nativo (ONNX Runtime + modelos) en models\onnx
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$rust = Join-Path $root "rust"
if (-not $TargetDir) { $TargetDir = Join-Path $rust "target-dist" }
if (-not $DistRoot) { $DistRoot = Join-Path $root "dist" }
$dist = Join-Path $DistRoot "Reescaller"

# --- versión desde Cargo.toml -------------------------------------------------
$version = (Select-String -Path (Join-Path $rust "Cargo.toml") -Pattern '^version\s*=\s*"([^"]+)"' |
            Select-Object -First 1).Matches.Groups[1].Value
Write-Host "Reescaller $version" -ForegroundColor Cyan

# --- 1) compilar ---------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host "[1/5] Compilando en release..." -ForegroundColor Cyan
    $env:CARGO_TARGET_DIR = $TargetDir
    & cmd /c "`"$rust\build.bat`" build --release"
    if ($LASTEXITCODE -ne 0) { throw "La compilación falló." }
} else {
    Write-Host "[1/5] Compilación omitida (-SkipBuild)." -ForegroundColor Yellow
}

$rel = Join-Path $TargetDir "release"
foreach ($f in "reescaller-app.exe", "reescaller.exe") {
    if (-not (Test-Path (Join-Path $rel $f))) { throw "Falta $rel\$f. Compila primero." }
}

# --- 2) montar la carpeta ---------------------------------------------------------
Write-Host "[2/5] Montando $dist ..." -ForegroundColor Cyan
# Si Reescaller.exe está abierto no se toca nada (borraría el resto de archivos a medias).
$exeDest = Join-Path $dist "Reescaller.exe"
if (Test-Path $exeDest) {
    try { [IO.File]::Open($exeDest, "Open", "ReadWrite", "None").Close() }
    catch { throw "Reescaller.exe está en uso: ciérralo antes de empaquetar (o usa -DistRoot otra_carpeta)." }
}
# Se vacía el contenido (no la carpeta): puede estar abierta en el Explorador.
if (Test-Path $dist) { Get-ChildItem $dist -Force | Remove-Item -Recurse -Force }
New-Item -ItemType Directory -Force (Join-Path $dist "models\models") | Out-Null

Copy-Item (Join-Path $rel "reescaller-app.exe") (Join-Path $dist "Reescaller.exe")
# La ventana nativa (WebView2) necesita su cargador junto al ejecutable.
$registry = Join-Path (Join-Path $env:USERPROFILE ".cargo") "registry\src"
$loader = Get-ChildItem $registry -Recurse -Filter WebView2Loader.dll -ErrorAction SilentlyContinue |
          Where-Object { $_.Directory.Name -eq "x64" } | Select-Object -First 1
          Where-Object { $_.FullName -match "[\/]x64[\/]" } | Select-Object -First 1
if ($loader) {
    $dest = Join-Path $dist "WebView2Loader.dll"
    Copy-Item $loader.FullName $dest
    (Get-Item $dest).LastWriteTime = Get-Date   # el DLL trae una fecha anterior a 1980 que Zip no admite
}
else { Write-Warning "No se encontró WebView2Loader.dll (x64): la app usará Edge en modo ventana como respaldo." }
# El CLI no puede llamarse reescaller.exe: Windows no distingue mayúsculas.
Copy-Item (Join-Path $rel "reescaller.exe") (Join-Path $dist "reescaller-cli.exe")

$src = Join-Path $root "models"
foreach ($f in "realesrgan-ncnn-vulkan.exe", "ffmpeg.exe") {
    if (-not (Test-Path (Join-Path $src $f))) { throw "Falta models\$f" }
    Copy-Item (Join-Path $src $f) (Join-Path $dist "models\$f")
}
Copy-Item (Join-Path $src "models\*") (Join-Path $dist "models\models") -Include *.bin, *.param

# --- 3) runtime de Visual C++ ------------------------------------------------------
Write-Host "[3/5] Runtime VC++ (vcomp140.dll)..." -ForegroundColor Cyan
$vcomp = Join-Path $env:SystemRoot "System32\vcomp140.dll"
if (Test-Path $vcomp) {
    Copy-Item $vcomp (Join-Path $dist "models\vcomp140.dll")
} else {
    Write-Warning "No se encontró vcomp140.dll en este equipo: instala el Visual C++ Redistributable 2015-2022 (x64) y repite."
}

# --- 3b) motor de rostros nativo (opcional) ----------------------------------------
if ($WithOnnx) {
    Write-Host "[3b] Motor de rostros ONNX (models\onnx)..." -ForegroundColor Cyan
    $onnxSrc = Join-Path $root "experimental\onnx-faces"
    $onnxDst = Join-Path $dist "models\onnx"
    New-Item -ItemType Directory -Force $onnxDst | Out-Null
    foreach ($f in "onnxruntime.dll", "DirectML.dll", "onnxruntime_providers_shared.dll") {
        Copy-Item (Join-Path $onnxSrc "bin\$f") $onnxDst
    }
    foreach ($f in "gfpgan_v14.onnx", "restoreformer_pp.onnx", "retinaface_r50.onnx", "parsenet.onnx") {
        $p = Join-Path $onnxSrc "models\$f"
        if (-not (Test-Path $p)) { throw "Falta $p (ejecuta experimental\onnx-faces\export_onnx.py)" }
        Copy-Item $p $onnxDst
    }
}

# --- 3b2) licencias: la de Reescaller, los avisos de terceros y los textos originales (obligatorio al redistribuir FFmpeg GPL, modelos, etc.) ---
Copy-Item (Join-Path $root "LICENSE") (Join-Path $dist "LICENSE.txt")
Copy-Item (Join-Path $root "THIRD_PARTY_NOTICES.md") (Join-Path $dist "THIRD_PARTY_NOTICES.md")
$licSrc = Join-Path $root "licenses"
if (-not (Test-Path (Join-Path $licSrc "RUST_CRATES.txt"))) { throw "Falta licenses\RUST_CRATES.txt (python -I scripts\gen_third_party.py)." }
Copy-Item $licSrc (Join-Path $dist "licenses") -Recurse

# --- 3c) quitar la marca «descargado de Internet» (evita el aviso de seguridad de Windows al lanzar el motor) ---
Get-ChildItem $dist -Recurse -File | Unblock-File

# --- 3d) firma de código (opcional; sin certificado configurado no hace nada; ver scripts\sign.ps1) ---
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "sign.ps1") (Join-Path $dist "Reescaller.exe") (Join-Path $dist "reescaller-cli.exe")
if ($LASTEXITCODE -ne 0) { throw "Falló la firma de los ejecutables." }

# --- 4) documentación y zip ----------------------------------------------------------
Write-Host "[4/5] Documentación y ZIP..." -ForegroundColor Cyan
@"
REESCALLER $version - remasterizacion de imagenes con IA
========================================================

USO
  Doble clic en Reescaller.exe. Se abre su propia ventana; al cerrarla, la app termina.
  No necesita instalacion ni permisos de administrador.

REQUISITOS
  - Windows 10/11 de 64 bits con Microsoft Edge (o Google Chrome).
  - Para la mejora por IA: una GPU con Vulkan (NVIDIA, AMD o Intel Arc/iGPU reciente)
    y el driver grafico actualizado. Sin GPU compatible funciona en modo CPU (sin IA).

CONFIGURACION Y DATOS (por usuario, no se escribe en esta carpeta)
  %APPDATA%\Reescaller\config.json       ajustes
  %LOCALAPPDATA%\Reescaller\             logs, perfil de la ventana y subidas temporales

LINEA DE COMANDOS
  reescaller-cli.exe doctor              diagnostico del equipo
  reescaller-cli.exe --help

LICENCIAS
  - Reescaller: MIT (LICENSE.txt).
  - Componentes de terceros (FFmpeg GPL v3, Real-ESRGAN, GFPGAN, RestoreFormer++, ONNX Runtime, DirectML, etc.):
    ver THIRD_PARTY_NOTICES.md y la carpeta licenses\ . IMPORTANTE: los modelos de rostros pueden tener
    restricciones de uso comercial (detalle en THIRD_PARTY_NOTICES.md).
"@ | Set-Content -Encoding UTF8 (Join-Path $dist "LEEME.txt")

if ($NoZip) {
    Write-Host "      ZIP omitido (-NoZip)." -ForegroundColor Yellow
} else {
    $zip = Join-Path $DistRoot "Reescaller-$version-win64.zip"
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path $dist -DestinationPath $zip -CompressionLevel Optimal
    $mb = [math]::Round((Get-Item $zip).Length / 1MB, 1)
    Write-Host "      ZIP: $zip ($mb MB)" -ForegroundColor Green
}

# --- 5) instaladores ------------------------------------------------------------------
# a) Instalador propio (ventana con presentación de funciones y barra de progreso): dist\Reescaller-<version>-Setup.exe
Write-Host "[5/5] Instalador propio..." -ForegroundColor Cyan
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_setup.ps1") -TargetDir $TargetDir -DistRoot $DistRoot -SkipBuild
if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el instalador propio." }

# b) Alternativa clásica con Inno Setup (opcional): dist\Reescaller-<version>-Setup-Inno.exe
Write-Host "      Instalador alternativo (Inno Setup)..." -ForegroundColor Cyan
# Bootstrapper de Microsoft Edge WebView2 (lo instala el instalador si el equipo no lo tiene). Se verifica su firma.
$wv2 = Join-Path $root "installer\redist\MicrosoftEdgeWebview2Setup.exe"
if (-not (Test-Path $wv2)) {
    New-Item -ItemType Directory -Force (Split-Path $wv2) | Out-Null
    Write-Host "      Descargando el bootstrapper de WebView2..."
    Invoke-WebRequest -UseBasicParsing "https://go.microsoft.com/fwlink/p/?LinkId=2124703" -OutFile $wv2
}
$firma = Get-AuthenticodeSignature $wv2
if ($firma.Status -ne "Valid" -or $firma.SignerCertificate.Subject -notmatch "Microsoft Corporation") {
    throw "El bootstrapper de WebView2 no tiene una firma válida de Microsoft."
}
$iscc = @(
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($iscc) {
    & $iscc "/DAppVersion=$version" (Join-Path $root "installer\Reescaller.iss")
    if ($LASTEXITCODE -ne 0) { throw "Inno Setup falló." }
    Write-Host "      Instalador creado en dist\" -ForegroundColor Green
} else {
    Write-Host "      Inno Setup no está instalado: se omite. (winget install JRSoftware.InnoSetup)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Listo: $dist" -ForegroundColor Green
