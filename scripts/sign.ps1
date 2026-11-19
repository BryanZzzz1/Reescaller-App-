<#
.SYNOPSIS
  Firma ejecutables con Authenticode (SHA-256 + marca de tiempo) si hay un certificado configurado; si no, no hace nada.

.DESCRIPTION
  Configura UNA de estas opciones en variables de entorno (nunca en archivos del repositorio):
    REESCALLER_SIGN_PFX + REESCALLER_SIGN_PASSWORD   certificado .pfx y su contraseña
    REESCALLER_SIGN_THUMBPRINT                       huella de un certificado instalado en el almacén del usuario o del equipo
  Opcional: REESCALLER_SIGN_TIMESTAMP (por defecto http://timestamp.digicert.com).
  Sin certificado devuelve éxito y avisa; así el empaquetado funciona igual en equipos sin firma.

.EXAMPLE
  $env:REESCALLER_SIGN_PFX = "C:\certs\reescaller.pfx"; $env:REESCALLER_SIGN_PASSWORD = "..."
  powershell -File scripts\sign.ps1 dist\Reescaller\Reescaller.exe
#>
param([Parameter(Mandatory, ValueFromRemainingArguments)][string[]]$Path)

$ErrorActionPreference = "Stop"
$cert = $null
if ($env:REESCALLER_SIGN_PFX) {
    $pwd = if ($env:REESCALLER_SIGN_PASSWORD) { ConvertTo-SecureString $env:REESCALLER_SIGN_PASSWORD -AsPlainText -Force } else { $null }
    $cert = Get-PfxCertificate -FilePath $env:REESCALLER_SIGN_PFX -Password $pwd
} elseif ($env:REESCALLER_SIGN_THUMBPRINT) {
    $cert = Get-ChildItem Cert:\CurrentUser\My, Cert:\LocalMachine\My -ErrorAction SilentlyContinue |
            Where-Object { $_.Thumbprint -eq $env:REESCALLER_SIGN_THUMBPRINT } | Select-Object -First 1
    if (-not $cert) { throw "No se encontró el certificado con huella $($env:REESCALLER_SIGN_THUMBPRINT)." }
}
if (-not $cert) {
    Write-Host "      (sin certificado de firma: $($Path.Count) archivo(s) quedan sin firmar; ver ROADMAP.md)" -ForegroundColor Yellow
    return
}
$ts = if ($env:REESCALLER_SIGN_TIMESTAMP) { $env:REESCALLER_SIGN_TIMESTAMP } else { "http://timestamp.digicert.com" }
foreach ($p in $Path) {
    $r = Set-AuthenticodeSignature -FilePath $p -Certificate $cert -HashAlgorithm SHA256 -TimestampServer $ts
    if ($r.Status -notin "Valid", "UnknownError") { throw "No se pudo firmar $p : $($r.Status) $($r.StatusMessage)" }
    Write-Host "      firmado: $(Split-Path $p -Leaf)" -ForegroundColor Green
}
