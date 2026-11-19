<#
.SYNOPSIS
  Crea el instalador propio de Reescaller (ventana con presentación de funciones y barra de progreso).

.DESCRIPTION
  Compila el crate reescaller-setup (stub) y le añade al final un ZIP con dist\Reescaller\ (la app ya empaquetada por
  build_dist.ps1), el desinstalador y el instalador de WebView2. Resultado: dist\Reescaller-<version>-Setup.exe

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\build_setup.ps1
  powershell -ExecutionPolicy Bypass -File scripts\build_setup.ps1 -SkipBuild -Out dist\prueba-Setup.exe
#>
param(
    [switch]$SkipBuild,
    [string]$TargetDir = "",
    [string]$DistRoot = "",
    [string]$Out = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$rust = Join-Path $root "rust"
if (-not $TargetDir) { $TargetDir = Join-Path $rust "target-dist" }
if (-not $DistRoot) { $DistRoot = Join-Path $root "dist" }
$app = Join-Path $DistRoot "Reescaller"
$version = (Select-String -Path (Join-Path $rust "Cargo.toml") -Pattern '^version\s*=\s*"([^"]+)"' |
            Select-Object -First 1).Matches.Groups[1].Value
if (-not $Out) { $Out = Join-Path $DistRoot "Reescaller-$version-Setup.exe" }

if (-not (Test-Path (Join-Path $app "Reescaller.exe"))) { throw "Falta $app\Reescaller.exe: ejecuta antes scripts\build_dist.ps1." }

if (-not $SkipBuild) {
    Write-Host "[1/3] Compilando el instalador..." -ForegroundColor Cyan
    $env:CARGO_TARGET_DIR = $TargetDir
    & cmd /c "`"$rust\build.bat`" build --release -p reescaller-setup"
    if ($LASTEXITCODE -ne 0) { throw "La compilación del instalador falló." }
}
$launcher = Join-Path $TargetDir "release\reescaller-setup-launcher.exe"
$ui = Join-Path $TargetDir "release\reescaller-setup.exe"
foreach ($f in $launcher, $ui) { if (-not (Test-Path $f)) { throw "Falta $f" } }
# La interfaz (WebView2) necesita su cargador al lado: es el mismo que lleva la app.
$loader = Join-Path $app "WebView2Loader.dll"
if (-not (Test-Path $loader)) { throw "Falta $loader (lo copia scripts\build_dist.ps1)." }

Write-Host "[2/3] WebView2 (instalador oficial de Microsoft)..." -ForegroundColor Cyan
$wv2 = Join-Path $root "installer\redist\MicrosoftEdgeWebview2Setup.exe"
if (-not (Test-Path $wv2)) {
    New-Item -ItemType Directory -Force (Split-Path $wv2) | Out-Null
    Invoke-WebRequest -UseBasicParsing "https://go.microsoft.com/fwlink/p/?LinkId=2124703" -OutFile $wv2
}
$firma = Get-AuthenticodeSignature $wv2
if ($firma.Status -ne "Valid" -or $firma.SignerCertificate.Subject -notmatch "Microsoft Corporation") {
    throw "El instalador de WebView2 no tiene una firma válida de Microsoft."
}

Write-Host "[3/3] Empaquetando la app dentro del instalador..." -ForegroundColor Cyan
& python -I (Join-Path $PSScriptRoot "pack_setup.py") $launcher $ui $loader $app $wv2 $Out
if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el instalador." }
Write-Host "Listo: $Out" -ForegroundColor Green
