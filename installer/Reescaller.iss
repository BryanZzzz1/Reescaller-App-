; Instalador de Reescaller (Inno Setup 6)
; Compilar:  ISCC.exe /DAppVersion=0.3.0 installer\Reescaller.iss
; (scripts\build_dist.ps1 lo hace automáticamente si Inno Setup está instalado)
;
; · Instala por usuario (sin pedir administrador): {autopf} apunta a %LOCALAPPDATA%\Programs.
; · Compatible con Windows 10 (1809 o posterior) y Windows 11, 64 bits.
; · Instala por su cuenta lo que la app necesita y puede faltar:
;     - Microsoft Edge WebView2 Runtime (la ventana de la app). Viene con Windows 11 y con Edge actualizado.
;     - Avisa si el equipo no tiene un driver de GPU con Vulkan (sin él la app funciona, pero sin IA).
;   El motor de IA, FFmpeg, vcomp140.dll, ONNX Runtime y DirectML ya van dentro de la carpeta de la app.

#ifndef AppVersion
  #define AppVersion "0.3.0"
#endif

[Setup]
AppId={{6E7C2F0B-6B1F-4C55-9F0A-5A0C3C1D7E21}
AppName=Reescaller
AppVersion={#AppVersion}
AppPublisher=Reescaller
DefaultDirName={autopf}\Reescaller
DefaultGroupName=Reescaller
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist
OutputBaseFilename=Reescaller-{#AppVersion}-Setup-Inno
SetupIconFile=..\rust\crates\reescaller-app\assets\reescaller.ico
UninstallDisplayIcon={app}\Reescaller.exe
Compression=lzma2/normal
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
WizardStyle=modern dark
DisableWelcomePage=no
WizardImageFile=assets\wizard-164.bmp,assets\wizard-192.bmp,assets\wizard-246.bmp,assets\wizard-328.bmp
WizardSmallImageFile=assets\small-55.bmp,assets\small-64.bmp,assets\small-80.bmp,assets\small-110.bmp
CloseApplications=yes
RestartApplications=no
DiskSpanning=no

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "full"; Description: "Instalación completa (recomendada)"
Name: "light"; Description: "Instalación ligera (sin restauración de rostros)"
Name: "custom"; Description: "Personalizada"; Flags: iscustom

[Components]
Name: "main"; Description: "Reescaller (imágenes, video y motor de IA)"; Types: full light custom; Flags: fixed
Name: "faces"; Description: "Restauración de rostros (modelos ONNX, unos 800 MB)"; Types: full

[CustomMessages]
spanish.WelcomeTitle=Bienvenido a Reescaller
spanish.WelcomeText=Mejora tus imágenes y videos con inteligencia artificial: más nitidez, más detalle y rostros restaurados. Todo se procesa en tu equipo, sin subir nada a internet.%n%nEste asistente instalará Reescaller y lo que necesita para funcionar.
spanish.FinishTitle=¡Todo listo!
spanish.FinishText=Reescaller se instaló correctamente.%n%nArrastra una imagen a la ventana, pégala con Ctrl+V o elige un archivo para empezar.
english.WelcomeTitle=Welcome to Reescaller
english.WelcomeText=Enhance your images and videos with artificial intelligence: more sharpness, more detail and restored faces. Everything is processed on your computer, nothing is uploaded to the internet.%n%nThis wizard will install Reescaller and what it needs to run.
english.FinishTitle=All set!
english.FinishText=Reescaller was installed successfully.%n%nDrag an image onto the window, paste it with Ctrl+V or choose a file to get started.

[Tasks]
Name: "desktopicon"; Description: "Crear un acceso directo en el escritorio"; GroupDescription: "Accesos directos:"

#ifdef QUICK
; Prueba rápida del aspecto (no incluye la app): ISCC /DQUICK=1
[Files]
Source: "assets\*"; DestDir: "{app}"; Flags: ignoreversion; Components: main
#else
[Files]
Source: "..\dist\Reescaller\*"; DestDir: "{app}"; Excludes: "models\onnx\*"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: main
Source: "..\dist\Reescaller\models\onnx\*"; DestDir: "{app}\models\onnx"; Flags: ignoreversion recursesubdirs createallsubdirs; Components: faces
; Microsoft Edge WebView2 Runtime (bootstrapper firmado por Microsoft): solo si el equipo no lo tiene.
Source: "redist\MicrosoftEdgeWebview2Setup.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not WebView2Installed
#endif

[Icons]
Name: "{autoprograms}\Reescaller"; Filename: "{app}\Reescaller.exe"
Name: "{autodesktop}\Reescaller"; Filename: "{app}\Reescaller.exe"; Tasks: desktopicon

[Run]
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "Instalando Microsoft Edge WebView2 Runtime (necesario para la ventana de la app)..."; Flags: waituntilterminated; Check: not WebView2Installed
Filename: "{app}\Reescaller.exe"; Description: "Abrir Reescaller"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Los ajustes (%APPDATA%\Reescaller) se conservan; se borran la caché de la ventana y las imágenes subidas o pegadas.
Type: filesandordirs; Name: "{localappdata}\Reescaller\webview"
Type: filesandordirs; Name: "{localappdata}\Reescaller\webview2"
Type: filesandordirs; Name: "{localappdata}\Reescaller\uploads"

[Code]
const
  WebView2Guid = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

function WebView2Version(Root: Integer; const Key: string): string;
begin
  if not RegQueryStringValue(Root, Key, 'pv', Result) then
    Result := '';
end;

{ ¿Está instalado Microsoft Edge WebView2 Runtime (por equipo o por usuario)? }
function WebView2Installed: Boolean;
var
  v: string;
begin
  v := WebView2Version(HKLM, 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + WebView2Guid);
  if (v = '') or (v = '0.0.0.0') then
    v := WebView2Version(HKLM, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + WebView2Guid);
  if (v = '') or (v = '0.0.0.0') then
    v := WebView2Version(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\' + WebView2Guid);
  Result := (v <> '') and (v <> '0.0.0.0');
end;

procedure InitializeWizard;
begin
  WizardForm.WelcomeLabel1.Caption := CustomMessage('WelcomeTitle');
  WizardForm.WelcomeLabel2.Caption := CustomMessage('WelcomeText');
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then
  begin
    WizardForm.FinishedHeadingLabel.Caption := CustomMessage('FinishTitle');
    WizardForm.FinishedLabel.Caption := CustomMessage('FinishText');
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    { Sin Vulkan no hay motor de IA: se avisa para que el usuario actualice el driver de su GPU. }
    if not FileExists(ExpandConstant('{sys}\vulkan-1.dll')) then
      MsgBox('No se encontró un driver de GPU con Vulkan en este equipo.' + #13#10 + #13#10 +
             'Reescaller se instalará y funcionará, pero sin la mejora por IA hasta que instales o actualices ' +
             'el driver de tu tarjeta gráfica (NVIDIA GeForce GTX 600 o superior, AMD Radeon RX 400/500 o superior, Intel Arc).',
             mbInformation, MB_OK);
  end;
end;
