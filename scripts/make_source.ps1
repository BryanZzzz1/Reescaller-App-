<#
.SYNOPSIS
  Crea dist\Reescaller-<version>-source.zip con el código fuente (sin compilados, modelos ni entornos virtuales).

.DESCRIPTION
  Toma los archivos que Git versionaría (rastreados + nuevos que no estén en .gitignore) y los comprime con la carpeta raíz
  Reescaller-<version>-source/. Escribe también SHA256SUMS.txt en dist\ con las huellas de los entregables.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\make_source.ps1
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$dist = Join-Path $root "dist"
New-Item -ItemType Directory -Force $dist | Out-Null
$version = (Select-String -Path (Join-Path $root "rust\Cargo.toml") -Pattern '^version\s*=\s*"([^"]+)"' |
            Select-Object -First 1).Matches.Groups[1].Value

Push-Location $root
try {
    $files = git ls-files --cached --others --exclude-standard | Sort-Object -Unique |
             Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
} finally { Pop-Location }
if (-not $files) { throw "No se encontraron archivos (¿es un repositorio Git?)." }

$zipPath = Join-Path $dist "Reescaller-$version-source.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, "Create")
try {
    foreach ($f in $files) {
        $entry = "Reescaller-$version-source/" + ($f -replace '\\', '/')
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, (Join-Path $root $f), $entry, "Optimal")
    }
} finally { $zip.Dispose() }
$mb = [math]::Round((Get-Item $zipPath).Length / 1MB, 2)
Write-Host "Código fuente: $zipPath ($($files.Count) archivos, $mb MB)" -ForegroundColor Green

# Huellas de los entregables
$sums = Get-ChildItem $dist -File | Where-Object { $_.Name -match "^Reescaller-$([regex]::Escape($version)).*\.(exe|zip)$" } |
        ForEach-Object { "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name }
$sums | Set-Content -Encoding ASCII (Join-Path $dist "SHA256SUMS.txt")
Write-Host "Huellas en dist\SHA256SUMS.txt"
