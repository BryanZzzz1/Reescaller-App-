<div align="center">

# Reescaller

**Mejora tus imágenes y videos con inteligencia artificial. Todo en tu equipo, sin subir nada a internet.**

![Versión](https://img.shields.io/badge/versi%C3%B3n-0.3.0-8b5cf6?style=for-the-badge)
![Rust](https://img.shields.io/badge/Rust-1.75+-dea584?style=for-the-badge&logo=rust&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-10%20(1809)%20%7C%2011-0078d4?style=for-the-badge&logo=windows&logoColor=white)

</div>

Reescaller es una aplicación de escritorio para Windows escrita en **Rust**. Reescala y mejora fotos con **Real-ESRGAN**, restaura retratos completos (rostro, pelo, ropa y fondo) con **GFPGAN / RestoreFormer++** ejecutados de forma nativa con ONNX Runtime, y procesa video fotograma a fotograma con **FFmpeg**. La interfaz es una ventana propia con estilo de cristal, temas y 10 idiomas.

> Esta es la versión en Rust. La aplicación original en Python/CustomTkinter sigue en el repositorio como código legado ([ver más abajo](#código-legado-en-python)).

---

## Contenido

- [Características](#características)
- [Instalación](#instalación)
- [Requisitos](#requisitos)
- [Uso](#uso)
- [Compilar desde el código fuente](#compilar-desde-el-código-fuente)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Documentación](#documentación)
- [Estado y limitaciones](#estado-y-limitaciones)
- [Código legado en Python](#código-legado-en-python)

---

## Características

- **Imágenes**: escala libre de 0,25× a 8×, modos «siempre», «solo si hace falta» y «solo mejorar», mezcla de mejora 0–100 %, nitidez extra y peso de salida por calidad fija o por **tamaño objetivo en KB**. Respeta la orientación EXIF de las fotos de móvil.
- **Retrato completo**: restaura el rostro y mejora también pelo, ropa y fondo. Amplía automáticamente ×4 las fotos pequeñas, aclara las fotos oscuras, permite ajustar la fuerza de la restauración y trae un **comparador con deslizador y zoom**.
- **Video**: mejora fotograma a fotograma conservando el audio y el FPS originales; convierte a MP4, WebM, GIF o MP3.
- **Carpetas**: procesa lotes completos y guarda los resultados en una subcarpeta `Mejoradas` con el sufijo `_mejorado`.
- **Modo fácil y modo avanzado**: el modo fácil analiza la imagen y propone ajustes; el avanzado expone todos los parámetros.
- **Personalización**: modo claro/oscuro, ocho paletas, tema propio con selector de color, fondos animados, icono por tema y 10 idiomas.
- **Compatibilidad de hardware**: detecta CPU, RAM y GPU (NVIDIA GTX 600+, AMD RX 400+, Intel Arc) y avisa si falta un driver con Vulkan.
- **Privacidad**: todo se procesa localmente; la app solo escucha en `127.0.0.1` con un token por ejecución.
- **Instalador propio** con presentación de funciones y barra de progreso, instalación por usuario (sin administrador) y desinstalador.

## Instalación

1. Descarga `Reescaller-0.3.0-Setup.exe`.
2. Ejecútalo y pulsa **Aceptar e instalar**. En **Opciones** puedes cambiar la carpeta, omitir la restauración de rostros (≈ 800 MB) y decidir si se crea el acceso directo del escritorio.
3. Al terminar, pulsa **Abrir Reescaller**.

Se instala en `%LOCALAPPDATA%\Programs\Reescaller`, instala Microsoft Edge WebView2 Runtime si falta y se desinstala desde *Configuración → Aplicaciones instaladas*.

También hay una versión **portable** (`Reescaller-0.3.0-win64.zip`): descomprime y ejecuta `Reescaller.exe`.

Instalación silenciosa para scripts:

```bat
Reescaller-0.3.0-Setup.exe /S --dir="C:\Apps\Reescaller" [--no-faces] [--no-desktop]
"C:\Apps\Reescaller\Uninstall.exe" /S [--purge]
```

> El ejecutable aún **no está firmado**: Windows SmartScreen puede mostrar un aviso la primera vez.

## Requisitos

| | Mínimo |
|---|---|
| Sistema | Windows 10 (1809) o Windows 11, 64 bits |
| Ventana | Microsoft Edge WebView2 Runtime (el instalador lo añade si falta) |
| Mejora por IA | GPU con **Vulkan** y driver actualizado: NVIDIA GTX 600+, AMD RX 400+ o Intel Arc |
| Espacio | ≈ 1 GB con restauración de rostros; ≈ 200 MB sin ella |

Sin GPU compatible la app funciona, pero sin la mejora por IA. La restauración de rostros usa DirectML y se ha probado solo en AMD (ver [estado y limitaciones](#estado-y-limitaciones)).

## Uso

- Arrastra una imagen a la ventana, pégala con `Ctrl+V` o elige un archivo.
- **Imagen / Carpeta / Video / Rostros** están en la barra lateral; en el modo fácil basta con elegir un resultado y pulsar el botón.
- La configuración se guarda en `%APPDATA%\Reescaller\config.json`; los registros y la caché de la ventana, en `%LOCALAPPDATA%\Reescaller\`.

Herramientas de línea de comandos (carpeta de la app):

```bat
reescaller-cli.exe doctor
reescaller-cli.exe upscale --input foto.jpg --target 1.5 --weight-quality 88
reescaller-cli.exe batch --input entrada --output salida --target 2
```

## Compilar desde el código fuente

**Necesitas**: Rust 1.75+ con la toolchain `stable-x86_64-pc-windows-gnu` y [WinLibs MinGW](https://winlibs.com/) (`winget install BrechtSanders.WinLibs.POSIX.UCRT`). Opcional: Python 3 (empaquetado del instalador) e [Inno Setup 6](https://jrsoftware.org/isinfo.php) (instalador alternativo).

**Activos que no están en el repositorio** (carpeta `models/`, ≈ 800 MB, se distribuyen aparte):
`realesrgan-ncnn-vulkan.exe`, `ffmpeg.exe` y `models/*.bin|*.param`. Para los rostros en ONNX, ejecuta `experimental/onnx-faces/export_onnx.py`, que genera los modelos y librerías de `experimental/onnx-faces`.

> `rust/build.bat` contiene la ruta de WinLibs de la máquina del autor; ajusta la variable `MINGW` a tu instalación.

```bat
:: compilar todo
rust\build.bat build --release

:: comprobaciones (31 tests unitarios + verificador de entorno)
rust\build.bat test --workspace --release
probar.bat

:: carpeta portable + ZIP + instalador propio (+ Inno si está instalado)
crear_exe.bat                       :: equivale a scripts\build_dist.ps1
powershell -ExecutionPolicy Bypass -File scripts\build_dist.ps1 -WithOnnx
```

El resultado queda en `dist\`: `Reescaller\` (portable), `Reescaller-<ver>-win64.zip`, `Reescaller-<ver>-Setup.exe` y `Reescaller-<ver>-Setup-Inno.exe`.

Interfaz de depuración en el navegador (`http://127.0.0.1:7878`): `iniciar_debug.bat`.

## Estructura del repositorio

```
rust/                        workspace de Cargo
  crates/reescaller-core/    dominio: escalado, mezcla, peso, lote, video, rostros, hardware
  crates/reescaller-cli/     línea de comandos (reescaller-cli.exe)
  crates/reescaller-app/     aplicación de escritorio (Reescaller.exe) + servidor local + interfaz (index.html)
  crates/reescaller-selftest/ verificador de entorno
  crates/reescaller-setup/   instalador y desinstalador (lanzador + interfaz WebView2)
scripts/                     build_dist.ps1, build_setup.ps1, pack_setup.py, puente restore_face.py
installer/                   instalador alternativo (Inno Setup) y sus imágenes
experimental/onnx-faces/     exportación de los modelos de rostros a ONNX
docs/                        auditoría y ruta de migración nativa
LICENSE                      licencia MIT del código propio
THIRD_PARTY_NOTICES.md       licencias y avisos de los componentes de terceros
licenses/                    textos originales de esas licencias (+ RUST_CRATES.txt, generado)
PROJECT.md                   bitácora técnica detallada del proyecto
ROADMAP.md                   próximos pasos
All.py, requirements.txt     aplicación original en Python (legado)
```

## Documentación

- [`ROADMAP.md`](ROADMAP.md): qué falta y qué viene.
- [`PROJECT.md`](PROJECT.md): arquitectura, decisiones, pruebas y registro de cada ronda de cambios.
- [`docs/AUDITORIA.md`](docs/AUDITORIA.md) y [`docs/RUTA_MIGRACION_NATIVA.md`](docs/RUTA_MIGRACION_NATIVA.md).
- [`rust/README.md`](rust/README.md): detalle del workspace.

## Estado y limitaciones

- Versión **0.3.0**, en pruebas. El hardware probado es un PC con CPU Intel y GPU AMD RX 6600 (Windows 11): **NVIDIA, Intel Arc y portátiles con gráfica híbrida no se han probado**, ni Windows 10 en un equipo limpio.
- El ejecutable y el instalador **no están firmados** (el empaquetado ya admite firma: `scripts\sign.ps1`, ver [`ROADMAP.md`](ROADMAP.md)).
- Al guardar en PNG se pierde el canal alfa, y en lotes dos archivos con el mismo nombre y distinta extensión (`a.png` y `a.jpg`) pueden pisarse.
- Licencias de terceros revisadas con sus fuentes (no es asesoría legal): ver [Licencia](#licencia).

Si encuentras un problema, abre un *issue* indicando tu GPU, versión de Windows y el contenido de `reescaller-cli.exe doctor`.

## Código legado en Python

`All.py` es la aplicación original (CustomTkinter + PyTorch). Se conserva como referencia y no recibe novedades. Para ejecutarla: Python 3.10/3.11, `pip install -r requirements.txt`, `ffmpeg.exe`, `realesrgan-ncnn-vulkan.exe` y los pesos `GFPGANv1.4.pth` / `RestoreFormer++.ckpt` en `models/`, y luego `python All.py`.

## Licencia

El código de Reescaller es **MIT** ([`LICENSE`](LICENSE)). La aplicación incluye programas y modelos de terceros con sus propias licencias; el detalle, los textos originales y el código fuente de FFmpeg están en [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) y en [`licenses/`](licenses/) (también se instalan junto a la app).

Puntos a tener en cuenta:
- **FFmpeg** va como programa independiente bajo **GPL v3** (build de gyan.dev). Cada *release* adjunta el código fuente de FFmpeg 8.0.
- **Restauración de rostros**: GFPGAN y RestoreFormer++ se entrenaron con FFHQ, y GFPGAN incluye componentes de uso no comercial. Para uso personal no hay impedimento; para un uso **comercial** consulta antes `THIRD_PARTY_NOTICES.md`. El instalador permite omitir ese componente.
