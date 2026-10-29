; Inno Setup script for the Windows desktop application.
;
; The application itself is a one-folder PyInstaller build; speech engines are
; optional components copied from ``dist\components``. Compile with
; ``packaging\build_win.ps1``, which passes AppVersion, or manually:
;
;   iscc /DAppVersion=0.1.2 packaging\installer.iss

#define MyAppName "WakeWord Trainer"
#define MyAppPublisher "DUK-Codes"
#define MyAppUrl "https://github.com/DUK-Codes/wakeword-trainer"
#define MyAppExeName "WakeWordTrainer.exe"

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

; Folder produced by PyInstaller (dist\WakeWordTrainer next to this script).
#ifndef AppSource
  #define AppSource "..\dist\WakeWordTrainer"
#endif

#ifndef ComponentsSource
  #define ComponentsSource "..\dist\components"
#endif

[Setup]
AppId={{7F2B4E52-3A61-4C4B-9F2E-1E9C6B0D8A31}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppVerName={#MyAppName} {#AppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppUrl}
AppSupportURL={#MyAppUrl}/issues
AppUpdatesURL={#MyAppUrl}/releases
DefaultDirName={autopf}\WakeWord Trainer
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=wakeword-trainer-{#AppVersion}-windows-x64-setup
SetupIconFile=app.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "core"; Description: "Приложение / Application only"
Name: "withtts"; Description: "Приложение и движки озвучки / Application and speech engines"
Name: "custom"; Description: "Выборочная установка / Custom"; Flags: iscustom

[Components]
Name: "piper"; Description: "Piper — синтез речи, +45 МБ / speech engine"; Types: withtts

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#AppSource}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#ComponentsSource}\tts-piper\*"; DestDir: "{app}\components\tts-piper"; Components: piper; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove the engines that were installed together with the application. Files
; downloaded later by the Components page live in the user profile and are
; intentionally kept, so voices and negatives are not downloaded twice.
Type: filesandordirs; Name: "{app}\components"
