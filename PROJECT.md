# PROJECT.md — Herramienta de Mejora Visual (Reescaller-App)

> Documento técnico completo del proyecto, generado como base para su **remodelación**.
> Última revisión: sobre el commit `1016f20` ("Update README.md").

---

## 1. Resumen ejecutivo

| | |
|---|---|
| **Nombre** | Herramienta de Mejora Visual |
| **Tipo** | Aplicación de escritorio (GUI) para mejora de imágenes y video con IA |
| **Lenguaje** | Python 3.10+ |
| **UI** | CustomTkinter (tema dark/light) |
| **Plataforma** | Windows (binarios `.exe` embebidos) |
| **Núcleo de IA** | Real-ESRGAN (ncnn-vulkan, ejecutable externo), GFPGAN, RestoreFormer++ |
| **Multimedia** | FFmpeg (extracción/fusión de video, conversión, recorte) |
| **Entry point** | `All.py` (script único de ~760 líneas) |
| **Estado** | Funcional, monolítico. Requiere remodelación (ver §9). |

La aplicación permite:

1. Escalar/embellecer una imagen individual con Real-ESRGAN.
2. Mejorar todas las imágenes de una carpeta (procesamiento paralelo por hilos).
3. Restaurar rostros con GFPGAN o RestoreFormer++ (seleccionables).
4. Mejorar un video completo fotograma a fotograma y re-fusionar con el audio original.
5. Convertir y recortar videos con FFmpeg, con previsualización en la propia GUI.
6. Configurar rutas de ejecutables, factor de escala, modelo, formato de salida y tema.

---

## 2. Árbol de archivos

```
Reescaller-App-/
├── All.py                     # ← TODO el código de la aplicación (759 líneas)
├── README.md                  # Documentación de usuario (instalación/uso)
├── PROJECT.md                 # Este documento
├── .git/
└── models/                    # Recursos pesados + copia duplicada del entry point
    ├── All.py                 # ⚠ copia idéntica de ../All.py (35 KB)
    ├── ffmpeg.exe             # 95 MB  — FFmpeg portable
    ├── realesrgan-ncnn-vulkan.exe # 5.9 MB — motor de upscaling
    ├── GFPGANv1.4.pth         # 333 MB — pesos de restauración facial
    ├── RestoreFormer++.ckpt    # 281 MB — pesos de RestoreFormer++
    ├── models/                # 45 MB — realesr-animevideov3-x2 (.bin/.param)
    ├── RestoreFormer/         # código fuente del proyecto RestoreFormer upstream
    └── RestoreFormerPlusPlus/ # repo clonado de RestoreFormer++ (configs, scripts, gradio_demo)
```

**Observaciones de higiene del repo**

- `models/All.py` es un duplicado exacto de `All.py` → eliminar.
- `models/RestoreFormer/` y `models/RestoreFormerPlusPlus/` son **repos upstream completos** (con su propio `.git` en `RestoreFormerPlusPlus/`) → mover a `vendor/` o convertir en submódulos, no clutter.
- Binarios y checkpoints (~750 MB) están versionados en Git → mover a Git LFS o releases de GitHub.
- `models/models/` solo contiene `realesr-animevideov3-x2`, pero el modelo por defecto `realesrgan-x4plus` **no está presente**: el usuario debeproveerlo aparte.

---

## 3. Arquitectura actual

### 3.1 Diagrama

```
┌────────────────────────────────────────────────────────────┐
│                     All.py (monolito)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Variables    │  │ Estado global│  │  Funciones de   │  │
│  │ de ruta      │  │ rutas_selec. │  │  la GUI (build) │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Capa de Negocio (workers en threading.Thread)         │  │
│  │  • realesrgan_unico      • realesrgan_carpeta         │  │
│  │  • gdy/restoreformer     • proceso_video             │  │
│  │  • convertir_video       • recortar_video            │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────┐  ┌────────────────────────────┐  │
│  │ Helpers             │  │  Salida (UI)               │  │
│  │  • log_message       │  │  • log_text (CTkTextbox)   │  │
│  │  • set_progress      │  │  • barra_progreso          │  │
│  │  • proceso_ejecutar  │  │  • messagebox              │  │
│  └──────────────────────┘  └────────────────────────────┘  │
└───────┬───────────────────────┬────────────────────────────┘
        │ subprocess             │ import Python
        ▼                        ▼
  ffmpeg.exe            torch / basicsr / gfpgan
  realesrgan-ncnn-      RestoreFormer.RestoreFormer
  vulkan.exe
```

### 3.2 Inventario de APIs públicas de la UI

| Pestaña | Widgets / Variables |
|---|---|
| Imagen Única | `boton_seleccionar_unico`, `etiqueta_ruta_unico`, `boton_mejorar_unico` |
| Restaurar Rostros | `modelo_restauracion_var`, `modelo_restauracion_menu`, `boton_seleccionar_gfpgan` |
| Carpeta | `boton_seleccionar_entrada`, `etiqueta_entrada_carpeta`, `boton_seleccionar_salida`, `etiqueta_salida_carpeta`, `boton_mejorar_carpeta` |
| Video | `boton_seleccionar_video`, `etiqueta_ruta_video`, `boton_iniciar_video` |
| Convertir Video | `formato_video_var`, `formato_video_menu`, `boton_seleccionar_video_convertir`, `etiqueta_ruta_convertir`, `boton_convertir_video_iniciar` |
| Recortar Video | `etiqueta_visualizador`, `boton_seleccionar_video_recortar`, `etiqueta_ruta_recortar`, `boton_recortar_video_iniciar` |
| Configuración | `etiqueta_realesrgan/ffmpeg/gfpgan/restoreformer`, `escala_var`, `modelo_var`, `formato_var`, `eliminar_temp_var`, `tema_var` |
| Globales | `ventana`, `main_frame`, `notebook`, `barra_progreso`, `log_text` |

---

## 4. Flujos de trabajo detallados

### 4.1 Mejora de imagen individual
1. `seleccionar_archivo("imagen_unica")` → `filedialog.askopenfilename` → guarda en `rutas_seleccionadas`.
2. `ejecutar_realesrgan_unico()` valida `RUTA_EJECUTABLE` y la imagen.
3. En un hilo: construye `realesrgan-ncnn-vulkan.exe -i <in> -o <in>_mejorado.<fmt> -s <factor> -n <modelo>`.
4. `proceso_ejecutar()` ejecuta con `shell=True`, `capture_output=True`, `check=True`.
5. Log de éxito + `messagebox.showinfo`.

### 4.2 Mejora por carpeta
1. Se listan `*.png/*.jpg/*.jpeg` de la carpeta de entrada.
2. Se lanza un `ThreadPoolExecutor(max_workers=os.cpu_count())` con una tarea por archivo.
3. Cada tarea invoca `proceso_ejecutar(...)` con `paso_actual`/`total_pasos` para la barra de progreso.
4. El progreso se recalcula en el bucle de `futures`.

### 4.3 Restauración de rostros
1. `restaurar_cara()` abre el diálogo de selección y lee `modelo_restauracion_var`.
2. Delega en `ejecutar_gfpgan()` o `ejecutar_restoreformer()`.
3. Ambos cargan el modelo con PyTorch (`cuda` si está disponible, si no `cpu`),Calling `enhance(has_aligned=False, only_center_face=False)`.
4. Guardan `<nombre>_gfpgan.png` o `<nombre>_rf++.png` junto al original.
5. `ejecutar_gfpgan()` tiene auto-instalación de dependencias (`instalar_dependencias_pytorch()`).

### 4.4 Mejora de video (3 pasos)
1. **Extracción**: `ffmpeg -i video -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 tmp_frames/frame%08d.jpg`
2. **Mejora**: `realesrgan-ncnn-vulkan -i tmp_frames -o out_frames -n <modelo> -s <factor> -f jpg`
3. **Fusión**: `ffmpeg -i out_frames/frame%08d.jpg -i original -map 0:v:0 -map 1:a:0 -c:a copy -c:v libx264 -r 23.98 -pix_fmt yuv420p salida.mp4`
4. Limpieza opcional de `tmp_frames/` y `out_frames/`.

### 4.5 Conversión de video
`-vcodec copy -acodec copy` → remux sin recodificar. Prefijo `_convertido`.

### 4.6 Recorte de video
`ffmpeg -ss <inicio> -i video -t <duracion> -c copy salida.mp4`. Los tiempos se piden con `simpledialog.askstring`.

### 4.7 Previsualización (pestaña Recortar)
`cv2.VideoCapture` → lectura frame a frame en hilo daemon → conversión BGR→RGB → `PIL.Image` → resize LANCZOS a máx. 500×300 → `ctk.CTkImage` → `ventana.update()` en bucle.

---

## 5. Dependencias

### Python (pip)

| Paquete | Uso |
|---|---|
| `customtkinter` | Framework de UI |
| `Pillow` | Conversión de frames a imagen Tk |
| `opencv-python` (`cv2`) | I/O de imagen, `VideoCapture` |
| `numpy` | Arrays numéricos |
| `torch`, `torchvision` | Backend de GFPGAN / RestoreFormer++ |
| `basicsr` | `basicsr.utils.imwrite` + base de RestoreFormer |
| `gfpgan` | `GFPGANer` |
| `RestoreFormer` | Paquete local en `models/RestoreFormer` |

### Binarios externos

| Binario | Ruta esperada | Tamaño |
|---|---|---|
| FFmpeg | `models/ffmpeg.exe` | 95 MB |
| Real-ESRGAN ncnn-vulkan | `models/realesrgan-ncnn-vulkan.exe` | 5.9 MB |
| GFPGAN v1.4 | `models/GFPGANv1.4.pth` | 333 MB |
| RestoreFormer++ | `models/RestoreFormer++.ckpt` | 281 MB |
| realesr-animevideov3-x2 | `models/models/` | 45 MB |

### Requisitos implícitos

- Vulkan GPU (ncnn-vulkan) y/o CUDA para PyTorch.
- Python 3.10+ (basicsr y compilados de torchvision son sensibles a la versión).
-FFC en PATH si se decide no empaquetar `ffmpeg.exe`.

---

## 6. Interfaz de configuración actual

`class Config` (atributos de clase, se mutan con `setattr`):

```python
class Config:
    factor_escala   = "4x"      # 2x | 3x | 4x
    modelo          = "realesrgan-x4plus"
    formato_salida  = "png"     # png | jpg
    eliminar_temporales = True
    tema            = "Dark"    # Dark | Light
```

Rutas (globales mutables):
`RUTA_EJECUTABLE`, `RUTA_FFMPEG`, `RUTA_GFPGAN_PTH`, `RUTA_RESTOREFORMER_PTH`.

**Todos los ajustes son efímeros**: no se persisten entre ejecuciones. Cada arranque exige reconfigurar ejecutables manualmente.

---

## 7. Diagnóstico técnico — problemas detectados

### 7.1 Críticos

| # | Problema | Ubicación | Impacto |
|---|---|---|---|
| C1 | **`messagebox` y `update_idletasks()` desde hilos no-principales** | todas las funciones `worker()` | Comportamiento indefinido en Tkinter: la GUI puede congelarse o PET. `messagebox.showinfo` se llama desde el hilo worker. |
| C2 | **`shell=True` con lista de argumentos** | `proceso_ejecutar()` | En Windows `shell=True` con lista hace join ambiguo; combinado con rutas con espacios rompe la invocación. Debe ser `shell=False`. |
| C3 | **`-s Config.factor_escala[0]`** | `ejecutar_realesrgan_unico`, `ejecutar_realesrgan_carpeta`, `ejecutar_proceso_video` | Toma el **primer carácter** del string: `"4x"[0] == "4"` funciona por suerte, pero `"2x"[0] == "2"` también; frágil y no extensible (p. ej. `"1.5x"` → `"1"`). Debe ser `Config.factor_escala.rstrip("x")`. |
| C4 | **Frame rate fijo `-r 23.98`** | `ejecutar_proceso_video` | El video final queda desincronizado del original salvo que sea 23.98 fps. Debería heredarse del video de entrada. |
| C5 | **`except Exception: pass`** en varios workers | `ejecutar_realesrgan_unico`, `ejecutar_proceso_video`, `convertir_video` | Silencia errores reales: el usuario solo ve el log, nunca un aviso. |
| C6 | **Doble `import torch` / doble carga de modelo** | `ejecutar_gfpgan`, `ejecutar_restoreformer` | El modelo se carga en cada llamada; una sesión con varias imágenes reinstala/recarga pesos (~cientos de MB) cada vez. |
| C7 | **`importlib.reload(sys.modules['torch'])`** | `instalar_dependencias_pytorch()` | `reload` sobre módulos ya importados no "reinstala": lanza `ModuleNotFoundError` si no estaba importado, y no recarga sub-módulos (C extensions no se recargan). El flujo de auto-instalación está roto. |
| C8 | **Rutas de modelos por defecto inexistentes** | nivel raíz | `RUTA_GFPGAN_PTH` / `RUTA_RESTOREFORMER_PTH` arrancan vacías aunque los `.pth/.ckpt` están en `models/`: el usuario debe seleccionarlos a mano siempre. Debería autodetectarse. |

### 7.2 Altos

| # | Problema | Detalle |
|---|---|---|
| A1 | **Configuración no persistente** | `Config` y las 4 rutas no se guardan. Cada ejecución empieza desde cero. |
| A2 | **Mezcla de construcción de UI y lógica** | Todo el build de widgets está a nivel de módulo (líneas ~520–740), imposible de testear o reutilizar. |
| A3 | **`video_reproductor` no se libera al cambiar de vídeo de forma segura** | Se llama `release()` mientras el hilo de reproducción sigue leyendo → condiciones de carrera. |
| A4 | **Bucle de reproducción sin control de fin ni `after()`** | `while video_reproductor.isOpened()` + `ventana.update()` en un hilo secundario →人人 puede "comer" eventos de Tk, y no hay forma de detener la reproducción. |
| A5 | **`restore` no valida la existencia del modelo antes de descargar** | `RUTA_GFPGAN_PTH` vacío → `GFPGANer(model_path="")` falla con error poco claro. |
| A6 | **Sobrecarga de `ThreadPoolExecutor` con `os.cpu_count()`** | Lanza N procesos de Vulkan simultáneos → saturación de GPU/VRAM y timeouts. Debería limitarse (p. ej. 1–2) o configurable. |
| A7 | **La barra de progreso en modo carpeta se desincroniza** | Varias tareas escriben progreso en paralelo con índices compartidos → saltos. |
| A8 | **Sin manejo de cancelación** | Un video largo no se puede detener; FFmpeg se lanza como proceso no controlable. |
| A9 | **`simpledialog` en un hilo worker** | `recortar_video` pide input dentro del worker → diálogo no modal y OFF-THREAD. |
| A10 | **No hay verificación de dependencias al arranque** | `verificar_dependencias_inicio()` solo mira `RUTA_EJECUTABLE` y `RUTA_FFMPEG`; no valida `torch`, `basicsr`, `gfpgan`, ni la existencia real de los archivos. |

### 7.3 Medios / menores

| # | Problema | Detalle |
|---|---|---|
| M1 | `except ImportError: pass` a nivel de import | Oculta la causa raíz de las dependencias faltantes. |
| M2 | Sin `requirements.txt` / `pyproject.toml` | No hay reproducible build. |
| M3 | Sin tests | Cero cobertura. |
| M4 | Sin `.gitignore` (en raíz) | Risk de commitear `venv/`, `tmp_frames/`, `out_frames/`. |
| M5 | `models/RestoreFormerPlusPlus/.git` | Repo anidado dentro del repo principal. |
| M6 | `models/All.py` | Duplicado exacto del entry point. |
| M7 | Nombres inconsistentes | `pestaña_gfpgan` maneja también RestoreFormer; `ejecutar_gfpgan` vs `ejecutar_restoreformer` asimétricos. |
| M8 | Mezcla de idiomas | Código en español/inglés, comentarios en español, mensajes de UI en español. Estandarizar. |
| M9 | Salidas siempre junto al original | Sobrescribe/ensucia la carpeta del usuario; falta opción "elegir carpeta de salida". |
| M10 | Sin drag & drop | Selección por diálogo nativo; en una app de escritorio es una limitación UX notable. |
| M11 | `Config.formato_salida` no se aplica al video | Solo afecta imágenes. |
| M12 | Sin previsualización antes/después | El usuario no puede comparar el resultado. |
| M13 | Seguridad de `subprocess` | Con `shell=True` (C2) hay riesgo de inyección si una ruta trae metacaracteres. |
| M14 | `proceso_ejecutar` no tiene timeout | Un ejecutable colgado bloquea la GUI indefinidamente. |
| M15 | Sin logging a archivo | Solo se ve el log en pantalla; imposible depurar crashes. |
| M16 | Sin i18n | Textos hardcodeados. |

---

## 8. Arquitectura propuesta para la remodelación

### 8.1 Principios

1. **Separar UI de lógica**: la GUI solo orquesta; toda la lógica es testeable sin ventana.
2. **Capa de dominio** sin dependencias de Tkinter ni de frameworks pesados.
3. **Backends intercambiables** para Real-ESRGAN (exe / PyTorch), GFPGAN, RestoreFormer++.
4. **Persistencia de configuración** en JSON en `%APPDATA%` / `~/.config`.
5. **Procesos cancelables** y con timeout; FFmpeg y Real-ESRGAN siempre con `shell=False`.
6. **Logging estructurado** a archivo + consola de la UI.
7. **Empaquetado** con PyInstaller en un único ejecutable portable.

### 8.2 Estructura de carpetas propuesta

```
Reescaller-App-/
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── README.md
├── PROJECT.md
├── src/
│   └── reescaller/
│       ├── __init__.py
│       ├── __main__.py            # python -m reescaller
│       ├── app.py                 # bootstrap de la aplicación
│       ├── config.py              # dataclasses + persistencia JSON
│       ├── logging_setup.py       # logging a archivo + callbacks a UI
│       ├── constants.py           # rutas por defecto, extensiones, modelos
│       ├── core/                  # ← dominio puro, sin Tkinter
│       │   ├── types.py           # dataclasses: Job, JobResult, Progress, Options
│       │   ├── paths.py           # detección de binarios/modelos (autodetect)
│       │   ├── process_runner.py   # wrapper de subprocess (shell=False, timeout, cancel)
│       │   ├── imaging.py          # upscale/restore (Real-ESRGAN, GFPGAN, RestoreFormer++)
│       │   ├── video.py           # pipeline extract → enhance → mux (ffmpeg)
│       │   ├── jobs.py            # orquestación, ThreadPool, progreso, cancelación
│       │   └── errors.py
│       ├── ui/
│       │   ├── main_window.py
│       │   ├── views/
│       │   │   ├── image_tab.py
│       │   │   ├── batch_tab.py
│       │   │   ├── face_tab.py
│       │   │   ├── video_tab.py
│       │   │   ├── settings_tab.py
│       │   │   └── widgets.py     # log panel, progress bar, preview
│       │   └── theme.py
│       └── resources/
│           ├── vendors/           # ffmpeg.exe, realesrgan-ncnn-vulkan.exe
│           └── models/            # .pth / .ckpt / .param / .bin (Git LFS)
├── tests/
│   ├── unit/  test_process_runner.py, test_config.py, test_scales.py ...
│   └── integration/ test_video_pipeline.py
├── scripts/
│   ├── build_windows.ps1
│   └── download_assets.ps1
└── docs/
    ├── ARCHITECTURE.md
    └── ROADMAP.md
```

### 8.3 Modelo de dominio (borrador)

```python
# src/reescaller/core/types.py
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class Scale(str, Enum):
    X2 = "2"; X3 = "3"; X4 = "4"

class EnhanceModel(str, Enum):
    REALESRGAN_X4PLUS = "realesrgan-x4plus"
    REALESRGAN_X4PLUS_ANIME = "realesrgan-x4plus-anime"
    ANIMEVIDEO_V3 = "realesr-animevideov3"

@dataclass(frozen=True)
class EnhanceOptions:
    model: EnhanceModel = EnhanceModel.REALESRGAN_X4PLUS
    scale: Scale = Scale.X4
    output_format: str = "png"
    denoise: bool = False

@dataclass
class JobProgress:
    current: int = 0
    total: int = 0
    message: str = ""
    @property
    def fraction(self) -> float:
        return self.current / self.total if self.total else 0.0

@dataclass
class JobResult:
    ok: bool
    outputs: list[Path] = field(default_factory=list)
    error: str | None = None
```

### 8.4 Capa de procesos (punto crítico de la remodelación)

```python
# src/reescaller/core/process_runner.py
import shutil, subprocess
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str

class ProcessRunner:
    """Wrapper de subprocess: shell=False, timeout, streaming de stdout, cancelación."""

    def __init__(self, timeout: int = 3600, on_line=None):
        self.timeout = timeout
        self.on_line = on_line
        self._process: subprocess.Popen | None = None

    def run(self, cmd: list[str], cwd: Path | None = None) -> CommandResult:
        exe = cmd[0]
        if not (Path(exe).exists() or shutil.which(exe)):
            raise ExecutableNotFoundError(exe)
        self._process = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", shell=False,
        )
        try:
            out, _ = self._process.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.cancel()
            raise
        ...

    def cancel(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try: self._process.wait(timeout=5)
            except subprocess.TimeoutExpired: self._process.kill()
```

### 8.5 Servicios de dominio

```python
class RealEsrganBackend:
    """Envuelve realesrgan-ncnn-vulkan.exe. Escala: '-s', str(4) sin 'x'."""
    def enhance_image(self, src: Path, dst: Path, opts: EnhanceOptions) -> Path: ...

class GfpganBackend:
    """Carga perezosa y cacheada del modelo GFPGANer (device auto)."""
    def enhance(self, src: Path, dst: Path) -> Path: ...

class RestoreFormerBackend:
    def enhance(self, src: Path, dst: Path) -> Path: ...

class VideoPipeline:
    """extract → enhance → mux, heredando fps del video de entrada (-r no forzado)."""
    def run(self, src: Path, dst: Path, opts: EnhanceOptions, on_progress) -> Path: ...
```

### 8.6 Gestión de configuración persistente

```python
# src/reescaller/config.py
@dataclass
class AppConfig:
    realesrgan_exe: str | None = None
    ffmpeg_exe: str | None = None
    gfpgan_model: str | None = None
    restoreformer_model: str | None = None
    scale: str = "4x"
    model: str = "realesrgan-x4plus"
    output_format: str = "png"
    delete_temp: bool = True
    theme: str = "Dark"
    max_workers: int = 2

    # persiste a %APPDATA%/Reescaller/config.json
```

Con **autodetección** en el arranque: si `resources/vendors/ffmpeg.exe` existe, se rellena solo; idem para los modelos.

---

## 9. Hoja de ruta de remodelación

### Fase 0 — Base (1–2 días)
- [ ] Crear `pyproject.toml` + `requirements.txt` + `.gitignore`.
- [ ] Estructura `src/reescaller/`.
- [ ] Mover `models/` a `resources/` y `vendor/`; sacarlo de Git (LFS/releases).
- [ ] Eliminar `models/All.py` y el `.git` anidado de `RestoreFormerPlusPlus`.

### Fase 1 — Núcleo sin UI (3–5 días)
- [ ] `core/types.py`, `core/errors.py`.
- [ ] `core/process_runner.py` con `shell=False`, timeout, cancelación, streaming de log.
- [ ] `core/paths.py` con autodetección de binarios/modelos.
- [ ] `core/imaging.py` con backends Real-ESRGAN / GFPGAN / RestoreFormer++ (modelo cacheado).
- [ ] `core/video.py` con pipeline de 3 pasos y fps heredado.
- [ ] `core/jobs.py` con cola, progreso agregado y cancelación.
- [ ] Tests unitarios de `process_runner`, `paths`, `video`.

### Fase 2 — Configuración y logging (1–2 días)
- [ ] `config.py` con dataclasses + persistencia JSON.
- [ ] `logging_setup.py`: logger a archivo +Bridge a la TextBox de la UI (thread-safe vía `after`).
- [ ] Migrar `class Config` → `AppConfig` y eliminar variables globales mutables.

### Fase 3 — Reconstrucción de la UI (3–5 días)
- [ ] `ui/main_window.py` con las 7 pestañas.
- [ ] Un widget = un archivo en `ui/views/`.
- [ ] Corregir C1: todo `messagebox` y la actualización de widgets desde el hilo principal (`ventana.after`).
- [ ] Previsualización antes/después (comparador con slider).
- [ ] Drag & drop de archivos (`tkinterdnd2`).
- [ ] Controles de video: inicio, duración, formato, calidad.

### Fase 4 — Calidad de proceso (2–3 días)
- [ ] Botón **Cancelar** por job.
- [ ] Límite de workers configurable (por defecto 1–2 para Real-ESRGAN).
- [ ] Manejo de errores con mensajes accionables (no `pass`).
- [ ] Validación de dependencias al arranque con checklist visual.

### Fase 5 — Empaquetado y entrega (2–3 días)
- [ ] Script de build con PyInstaller (`--onedir` con assets externos).
- [ ] `download_assets.ps1` para descargar modelos pesados bajo demanda.
- [ ] README actualizado, CHANGELOG, licencia.

---

## 10. Backlog de correcciones puntuales (aplicables ya, sin Waiting por laOMS arquitectura nueva)

| ID | Corrección | Esfuerzo | Estado |
|---|---|---|---|
| F1 | `shell=True` → `shell=False` en `proceso_ejecutar` | 1 línea | ✅ Aplicada |
| F2 | `Config.factor_escala[0]` → `Config.factor_escala.rstrip("x")` | 1 línea | ✅ Aplicada |
| F3 | Quitar `except Exception: pass` y loguear el traceback | 5 min | ✅ Aplicada |
| F4 | Mover `messagebox` al hilo principal con `ventana.after` | 30 min | ✅ Aplicada |
| F5 | Heredar el fps del video en el paso de fusión (o usar `-vsync vfr`) | 15 min | ✅ Aplicada |
| F6 | Autodetectar `models/ffmpeg.exe`, `realesrgan-ncnn-vulkan.exe` y los checkpoints | 30 min | ✅ Aplicada |
| F7 | Persistir la configuración en `config.json` | 1 h | ✅ Aplicada |
| F8 | Cachear los modelos GFPGAN/RestoreFormer entre llamadas | 1 h | ✅ Aplicada |
| F9 | Limitar `ThreadPoolExecutor` a 2 workers por defecto | 10 min | ✅ Aplicada |
| F10 | Eliminar `models/All.py` | 1 min | ✅ Aplicada |
| F11 | `importlib.reload` → reintentar `import` tras instalar | 15 min | ✅ Aplicada |
| F12 | Añadir `.gitignore` y `requirements.txt` | 15 min | ✅ Aplicada |

### 10.1 Registro de la tanda de correcciones aplicadas

**Estado: aplicadas y verificadas sobre `All.py` (verificado con `py_compile` y pruebas
funcionales reales con los binarios de `models/`).**

| Cambio | Detalle |
|---|---|
| F1 | `proceso_ejecutar()` usa `shell=False`, `encoding="utf-8"`, `errors="replace"` y timeout opcional. Se añadió `-y` a FFmpeg (evita que FFmpeg se quede esperando la confirmación de sobrescritura). |
| F2 | Nueva función `_escala_numerica()` (`"4x"` → `"4"`) usada en los 3 puntos donde se invocaba Real-ESRGAN. |
| F3 | Todos los `except: pass` eliminados; ahora se registra el error, se pone la barra a 0 y se muestra un `messagebox` al usuario. |
| F4 | Helpers `mostrar_info()` / `mostrar_error()` y `_en_hilo_principal()`: todos los `messagebox`, `log_message()`, `set_progress()` y `toggle_buttons_state()` se agendan en el hilo principal de Tk. Los diálogos de `simpledialog` en "Recortar Video" se piden **antes** de lanzar el worker. |
| F5 | Nueva función `detectar_fps()` (parsea la salida de FFmpeg) y uso de `-r <fps real>` + `-map 1:a:0?` (audio opcional) al fusionar. Verificado: entrada 25 fps → salida 25 fps con audio intacto. |
| F6 | `autodetectar_rutas()` busca ejecutables y checkpoints en `models/` y en la raíz; se ejecuta al arrancar y al reflejar las etiquetas de la pestaña Configuración. |
| F7 | `Config.save()` / `Config.load()` sobre `%APPDATA%/Reescaller/config.json`. Se guarda al cambiar cualquier opción, al configurar rutas y al cerrar la ventana. |
| F8 | Caché `_MODELOS_CARGADOS`: GFPGAN y RestoreFormer++ se cargan una sola vez por sesión. |
| F9 | `Config.max_workers` (nuevo control en Configuración, por defecto 2) limita el `ThreadPoolExecutor`. |
| F10 | `models/All.py` eliminado. |
| F11 | `instalar_dependencias_pytorch()` usa `importlib.import_module()` tras instalar (nada de `reload`, que no funciona con torch). |
| F12 | `requirements.txt` y `.gitignore` creados. |

**Extras añadidos por ser necesarios para "un entorno adecuado":**

| Extra | Detalle |
|---|---|
| `venv/` con Python 3.11 | `basicsr`/`gfpgan` no compilan en Python 3.14 (`KeyError: '__version__'`). El venv 3.11 tiene torch 2.15 CPU, torchvision 0.29, basicsr 1.4.2, gfpgan 1.3.8. |
| Parche de `torchvision` | `_parche_torchvision()` recrea `torchvision.transforms.functional_tensor` (eliminado en torchvision ≥ 0.17), necesario para importar `basicsr`/`gfpgan`/`facexlib`. Se aplica al importar el módulo. |
| `models/` en `sys.path` | Permite `import RestoreFormer.RestoreFormer` sin configurar nada. |
| `REESCALLER_HEADLESS=1` | Permite importar el módulo en pruebas automatizadas sin abrir el bucle principal. |
| Botón "Detener reproducción" | El visualizador de video ya no puede quedar bloqueado ni comerse eventos de Tk (A3/A4). |
| Verificación de dependencias al arranque | `verificar_dependencias_inicio()` muestra checklist de ejecutables, modelos y módulos de Python. |

**Verificaciones ejecutadas:**

```
✔ py_compile All.py                                  → OK
✔ Autodetección de 4 rutas (ffmpeg, realesrgan, GFPGAN, RestoreFormer++) → OK
✔ Guardado/carga de config.json                       → OK
✔ Real-ESRGAN con -s 4 (shell=False)                  → OK (rc=0, archivo generado)
✔ Pipeline de video completo (extract → enhance → mux) → 75 frames, salida 25 fps + audio 44100 Hz
✔ GFPGAN: carga de GFPGANv1.4.pth + enhance()         → OK
✔ RestoreFormer++: carga de RestoreFormer++.ckpt       → OK
✔ Arranque de la GUI (Python 3.14 y venv 3.11)        → sin tracebacks
```

---

## 11. Guía de desarrollo actual

```powershell
# Entorno
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt   # (a crear; ver Fase 0)

# Ejecutar
python All.py

# Assets requeridos en models/
#   ffmpeg.exe
#   realesrgan-ncnn-vulkan.exe
#   GFPGANv1.4.pth
#   RestoreFormer++.ckpt
#   models/realesr-animevideov3-x2.{bin,param}
```

**Depuración**

```powershell
python -X faulthandler All.py
python All.py 2>&1 | Tee-Object -FilePath run.log
```

---

## 12. Glosario

| Término | Significado |
|---|---|
| **Real-ESRGAN** | Modelo de super-resolución. `realesrgan-ncnn-vulkan` es su implementación en Vulkan (ejecutable autónomo). |
| **GFPGAN** | *Generative Facial Prior GAN*. Restauración de rostros sobre fotos degradadas. |
| **RestoreFormer++** | Restauración de rostros con transformador de alta calidad. |
| **ncnn-vulkan** | Runtime de inferencia sobre GPU Vulkan; no requiere PyTorch. |
| **basicsr** | Librería base de restauración de imagen; aporta `imwrite` y el andamiaje de RestoreFormer. |
| **Upscale / x4** | Multiplicar la resolución por 4 en cada eje. |
| **Mux/Fusionar** | Volver a unir video (frames mejorados) + audio original en un MP4 final. |
| **h.264 / libx264** | Códec de video usado en el paso de fusión. |
| **yuv420p** | Formato de píxel compatible con Reproductor y reproductores móviles. |

---

## 13. Riesgos y preguntas abiertas

| # | Pregunta | Impacto en la remodelación |
|---|---|---|
| 1 | ¿Se mantiene Windows como única plataforma, o se añade soporte Linux/macOS (cambia el empaquetado de binarios)? | Alcance de Fase 5 |
| 2 | ¿El motor Real-ESRGAN debe seguir siendo el `.exe` de Vulkan, o se migra a PyTorch (pip install realesrgan) para unificar el stack? | Decisión de arquitectura clave |
| 3 | ¿Se requiere modo CLI/headless además de la GUI? | Diseño de `core/` reutilizable |
| 4 | ¿Habrá modelo de licencia (comercial vs(open-source)?) | Decisión sobre redistribuir checkpoints en el repo |
| 5 | ¿SeIntegrará alguna función de IA generativa adicional (CodeFormer, realesr-anime, etc.)? | Extensibilidad de `core/imaging.py` |
| 6 | ¿La app necesita preferencia de usuario (login/sync) o es estrictamente local? | Alcance general |

---

## 14. Resumen ejecutivo para la dirección

La aplicación **funciona y es útil**, pero es un **script único de 760 líneas** que mezcla construcción de interfaz, lógica de negocio, invocación de procesos y carga de modelos de IA. El principal riesgo no es funcional sino **de mantenibilidad**: cualquier cambio requiere editar las mismas 760 líneas, es imposible testear sin abrir la ventana, y hay al menos 8 defectos críticos o altos (hilos Tkinter, `shell=True`, factor de escala truncado a un carácter, fps fijo, modelos recargados en cada llamada, auto-instalación rota, configuración no persistente y rutas sin autodetección).

La remediación propuesta es incremental y de bajo riesgo: **primero corregir los 12 defectos puntuales de la sección 10** (≈ 4 horas de trabajo) para estabilizar la base, y después ejecutar las fases **Fase 0 → Fase 5** que llevan el código a una arquitectura por capas (`core/` sin UI + `ui/` por vistas), con configuración persistente, procesos cancelables, tests y empaquetado portable. El resultado sería una aplicación mantenible, testeable y distribuible como ejecutable único.

---

## 15. Port a Rust (implementado)

La app Python (CustomTkinter) se ha portado a **Rust** en `rust/`, resolviendo el problema que el usuario disparó: el motor siempre devolvía la máxima resolución posible. Ahora se puede **elegir cuándo y cuánto** reescalar y definir el **peso** del archivo final.

### 15.1 Arquitectura del port

| Crate | Responsabilidad |
|---|---|
| `reescaller-core` | Dominio puro (sin UI): `config`, `process`, `image_ops`, `upscale`, `batch`, `video`, `face`, `diagnostics` |
| `reescaller-cli` | Binario `reescaller`: doctor, config, upscale, batch, video, face |
| `reescaller-app` | `Reescaller.exe` (ventana propia, sin consola) y `reescaller-debug.exe` (consola + navegador): servidor local (tiny_http) + UI embebida en `assets/index.html` |
| `reescaller-selftest` | Verificador automático: 18 comprobaciones PASS/FAIL del entorno y de todas las funciones (imagen, peso, lote, video, interfaz). Exit code 0/1 para CI |

### 15.2 Control del reescalado

```
original (w0 x h0)
  └─ motor Real-ESRGAN (x2 o x4, según modelo)      -> (w0*S, h0*S)
      └─ resize al objetivo (target_scale)           -> (w0*target, h0*target)
          └─ blend con la original (improve_percent) -> mezcla 0..100 %
              └─ sharpen (unsharp mask)              -> nitidez extra
                  └─ encode con Weight               -> calidad fija o peso objetivo
```

- `ResizeMode::Always` — aplica siempre `target_scale`.
- `ResizeMode::OnlyIfNeeded` — solo escala si el original es menor que el objetivo.
- `ResizeMode::ImproveOnly` — mejora la calidad conservando el tamaño original.
- `Weight::Quality { quality }` — calidad de codificación (JPEG 1–100; PNG = nivel de compresión).
- `Weight::TargetSize { target_kb }` — búsqueda binaria de la calidad que más se acerca al tamaño objetivo.

### 15.3 Toolchain en Windows

No hay Visual Studio Build Tools en la máquina, por lo que se usa el toolchain **GNU** de Rust con MinGW-w64 (WinLibs). Problema encontrado: **Git Bash (MSYS) corrompe el `PATH`** al pasarlo a procesos nativos, por lo que `rustc` no encontraba `dlltool.exe` y fallaba con `error calling dlltool 'dlltool.exe': program not found`.

Solución en `rust/build.bat` y `rust/build.sh`: reconstruir el `PATH` en formato Windows nativo (`MSYS2_ENV_CONV_EXCL="*"`) antes de invocar a `cargo`.

### 15.4 Verificaciones realizadas

```
✔ cargo build / build --release                     -> OK (2m02s en release)
✔ doctor                                             -> 7/7 comprobaciones OK
✔ upscale x1.5  (motor x4 -> 600x450)                -> exacto
✔ upscale x1.75 con peso objetivo 25 KB             -> 700x525, 24.1 KB
✔ upscale x1.25 con peso objetivo 15 KB (release)   -> 500x375, 15.1 KB en 3,5 s
✔ modo improve-only                                 -> conserva 400x300
✔ modo only-if-needed                                -> conserva 400x300
✔ batch de 7 imagenes (2 workers)                   -> 7/7 OK
✔ video (50 frames, escala 1.5)                     -> 480x360, 25 fps, audio 44100 Hz
✔ rostros GFPGAN via puente Python                  -> 800x600
✔ API HTTP de la interfaz debug                     -> upscale/batch/video/face/preview/upload/browse/doctor
✔ UI renderizada en navegador                        -> 5 pestanas operativas
✔ cargo test                                         -> 11 tests OK
✔ reescaller-selftest (verificador automatico)       -> 18/18 PASS
```

### 15.5 Interfaz de depuracion: lanzador y boton de ejecucion

Problemas resueltos:

1. **`iniciar_debug.bat` se abria y se cerraba al instante.** Causas: (a) la
   instancia anterior seguia occupying el puerto 7878, el servidor moria con
   error y la ventana se cerraba; (b) no habia `pause` al final, asi que
   cualquier salida cerraba la consola. Solucion: el lanzador cierra las
   instancias anteriores (`taskkill /F /IM reescaller-debug.exe`), comprueba el
   puerto con `netstat`, avisa si esta ocupado, ejecuta el servidor en primer
   plano (logs visibles), abre el navegador 3 s despues y hace `pause` al
   terminar. Ademas acepta `--debug` y `--port N`.
2. **La pestana Imagen no tenia un boton visible para ejecutar la mejora.**
   Ahora hay una barra de accion *sticky* arriba con `🚀 Procesar imagen`, un
   boton en la tarjeta *Comparativa* y el boton original dentro del plan.
   Durante el proceso se muestra un spinner y se deshabilitan los botones; al
   terminar aparece un aviso con las medidas y el peso del resultado. Atajo
   `Ctrl + Enter`.

### 15.6 Verificador automatico

`probar.bat` (raíz del proyecto) compila si hace falta y ejecuta
`reescaller-selftest.exe`, que comprueba en el equipo local:

```
entorno · Real-ESRGAN / FFmpeg / GFPGAN / RestoreFormer++ / puente Python / modelos ncnn
test · imagen de muestra generada (600x400)
reescalado · escala elegida x1.5 -> 900x600 exacto (no la maxima del motor)
reescalado · modo "solo mejora" -> conserva 600x400
peso · objetivo 30 KB -> 30.0 KB en 1200x800
reescalado · modo rapido sin motor IA -> 1200x800
lote · carpeta de 2 imagenes -> 2 correctas, 0 fallidas
video · escala 1.5 en frames -> 480x360, 25 fps heredados, audio conservado
interfaz · servidor de depuracion respondiendo
```

Bug de Windows documentado: `%~dp0` en un `.bat` termina en `\` y al
concatenarse con comillas (`"--project \"%~dp0\""`) produce rutas corruptas
(`...App-"`). Se soluciona usando `"%~dp0."` (misma carpeta, sin barra final)
en `cd /d` y en los argumentos.

### 15.7 Pendiente (ver también §16)

- GUI de escritorio nativa en Rust (egui/eframe) para sustituir la de CustomTkinter.
- Exportacion de GFPGAN/RestoreFormer++ a ONNX (`ort`) para eliminar el puente Python.
- Visualizador de video con preview en la GUI nativa.
- Tests del CLI (`cargo test --test cli`) y empaquetado con PyInstaller-equivalente (cargo-bundle /installer).

---

## 16. App de escritorio (.exe), interfaz nueva y compatibilidad de hardware

Resumen: la app Rust ahora es un **ejecutable único con ventana propia sin marco** (`Reescaller.exe`), con una
**interfaz mini** tipo chat para remasterizar/convertir en tres clics y una **interfaz avanzada** con todos los ajustes,
detección de hardware reforzada, temas (Clásico, Miami, Esmeralda, Zafiro, Magma y Distrito Neón, cada uno en claro/oscuro), guía animada y empaquetado portable
(con script para instalador).

### 16.1 Estado encontrado al empezar
- El crate de la interfaz **no compilaba** (`abrir_navegador` y `abrir_en_navegador` sin definir).
- Bug de CSS heredado: `rgba(var(--surface), .09)` con variables en formato `255 255 255` es sintaxis inválida, así
  que fondos, bordes de cristal y botones no se pintaban. Corregido con `rgb(var(--x) / alfa)`.
- El panel de logs ya no existía y `init()` dependía de controles aún no montados (`Cannot set properties of null`).
- La detección de GPU dependía de `vulkaninfo` (no viene en un PC limpio) y el motor necesita `VCOMP140.DLL`.

### 16.2 Ejecutable y ventana
| | |
|---|---|
| Binario | `Reescaller.exe` (crate `reescaller-app`, bin `reescaller-app`, subsistema Windows: sin consola). También `reescaller-debug.exe` (con consola) y `reescaller-cli.exe` (CLI; en `dist` se renombra para no chocar con `Reescaller.exe`, Windows no distingue mayúsculas) |
| Ventana | Nativa **sin marco** (`tao` + WebView2, módulo `native.rs`). Minimizar/maximizar/cerrar y la zona de arrastre los dibuja la UI y llegan por IPC (`drag`, `min`, `max`, `close`, `size:W,H,R`). Necesita `WebView2Loader.dll` junto al .exe (lo copia `build_dist.ps1`) y el runtime WebView2 (incluido en Windows 11). Si falla: Edge/Chrome en modo `--app`, y si no, navegador predeterminado |
| Tamaños | **Modo simple: 840×620, fija** (sin redimensionar ni maximizar; el botón ▢ se oculta). **Modo avanzado: 1360×880, redimensionable** (mínimo 900×600). Siempre arranca en mini (no se recuerda el último modo) |
| Cierre | Ventana nativa: al cerrarla termina el proceso. Respaldo Edge/Chrome: latido cada 3 s y cierre tras 25 s sin latido (sin trabajos en curso). No hay botón «Salir» |
| Instancia única | Puerto 7879: una segunda ejecución trae al frente la ventana abierta (`/api/focus`) |
| Datos | Ajustes en `%APPDATA%\Reescaller`; logs, subidas, perfil WebView2 y script del puente en `%LOCALAPPDATA%\Reescaller` (nada se escribe junto al .exe, apto para Program Files) |
| Seguridad | Solo `127.0.0.1`; se valida `Host` y toda llamada `/api/*` exige la cabecera `X-Reescaller` (sin CORS): evita CSRF/DNS-rebinding desde webs ajenas |
| Icono / logo | Marca propia: un píxel sólido y dos esquinas concéntricas que se expanden (idea de ampliar), sobre degradado violeta→cian. `assets/reescaller.ico` incrustado con `winres`; mismo símbolo en SVG dentro de la UI |

### 16.3 Interfaz
**Modo simple (por defecto)** — una tarjeta: zona grande para soltar/pegar/elegir imagen (comparador antes/después al
terminar) y barra inferior `[＋ Imagen · carpetas] [Acción · Tamaño] [Remasterizar]`. Acciones: Remasterizar
(Auto, Foto, Anime, Web ligera) o **Convertir** a PNG/JPG. Resultado con medidas, peso y botones Mostrar / Abrir / Otra.
Cabecera: pestañas **Imagen / Video / Rostros**, ayuda (?), claro/oscuro y **Modo avanzado**.

**Modo avanzado (vista completa)** — navegación:
| Entrada | Contenido |
|---|---|
| Remasterizar | Versión completa del modo fácil: pasos Imagen · Estilo · Tamaño e intensidad · Guardar en, y explicación del plan en vivo |
| Imagen y carpeta | Vista **unificada** (selector «Una imagen / Carpeta completa»). «Una imagen» reutiliza la vista de Remasterizar (entrada con Elegir/Pegar/Carpeta, arrastrar y pegar, comparador antes/después) con el plan completo debajo: escala del motor, escala final, cuándo reescalar, mejora %, peso (calidad o KB objetivo), nitidez, modelo, filtro, formato. «Carpeta completa» tiene su propio formulario |
| Video | Vista **unificada** (selector «Mejorar / Convertir formato»): formulario a la izquierda y panel de **Resultado** a la derecha |
| Restaurar rostros | Sin cambios (GFPGAN / RestoreFormer++ vía puente Python) |
| Ajustes | Pestañas **Equipo y GPU** · **Temas** · **Desarrollo** |

- «Modo simple» está junto al botón de tema en la barra superior. Quitados: indicadores de estado (motor/GPU/ffmpeg), botón
  «Salir» y «Recortar clip» (desactivado; el código sigue en el HTML). `lote`, `equipo`, `config` y `convertir` siguen
  funcionando como alias de las vistas nuevas.
- **Del modo simple al avanzado se conserva todo**: «Modo avanzado» carga la imagen (o carpeta), el resultado ya
  generado y los ajustes elegidos (modelo, escala, mejora, nitidez, formato, modo, calidad; sin IA si se estaba
  convirtiendo). El plan se obtiene con `POST /api/remaster {dry_run:true}`.
- **Temas**: dos paletas × claro/oscuro. *Clásico* (violeta/cian) y **Miami**, inspirado en el atardecer de neón de GTA VI:
  Deep Violet `#3B146B`, Purple `#7B208F`, Magenta `#C9349B`, Coral Pink `#F36A78`, Coral `#F87572`, Orange `#FF9B55`,
  Warm Yellow `#FFB458`. Se eligen en Ajustes → Temas; se guardan en `config.theme` (`miami-dark`, `classic-light`…) y
  en `localStorage`. Las variables CSS separan la superficie de las tarjetas (`--surface`) de la tinta de rellenos (`--ink`),
  lo que permite invertirla en claro.
- **Desarrollo**: editor de la configuración (`config.json`) con «Guardar cambios» (valida con `AppConfig`).
- **Equipo y GPU**: preferencia de dispositivo (auto/NVIDIA/AMD/Intel/CPU) o GPU concreta, hardware detectado, estado del
  entorno con «Informe» (copia diagnóstico) y tarjeta «Modelos y extras» (detectado/falta + «Elegir…» + «Buscar de nuevo»).
- **Componentes de UI propios**: desplegables con el estilo de la app (sustituyen a los `<select>` nativos; teclado,
  grupos, iconos, ✓), barra de scroll con degradado del tema, **iconos de trazo** que siguen el color del tema (sin
  emojis), avisos con icono, comparador antes/después, arrastrar y soltar y pegar con Ctrl+V.
- **Menos texto**: descripciones, pistas y avisos recortados; tarjetas con poco contenido fusionadas (sin «Pipeline»,
  «Plan aplicado» ni «Compatibilidad» sueltas).
- **Guía «Cómo se usa»** (botón ?): recorrido de 6 pasos con una escena **animada** por paso (soltar imagen, estilos,
  tamaño, comparador, mini→avanzado, video/ajustes), texto explicativo y consejo; avance automático cada 9 s con
  pausa, puntos, flechas y Esc.
- **Transiciones animadas**: entrada de vistas y tarjetas escalonadas, pestañas y selectores internos, mini↔completo,
  chips de resultado, avisos y modales. Respeta «reducir animaciones» del sistema.
- **«Mostrar en carpeta»** corregido: `explorer.exe /select,"ruta"` con `raw_arg` y barras normalizadas (antes, con
  espacios o paréntesis en la ruta, abría «Documentos»). Verificado: abre la carpeta con el archivo seleccionado.

### 16.4 Modo fácil (backend)
`core/src/remaster.rs`: analiza la imagen (tamaño, densidad de color → foto vs ilustración) y construye el `UpscalePlan`.
Presets Automático · Foto · Anime/dibujo · Web ligera; tamaño Auto/Mismo/×1.5–×4; intensidad Suave/Normal/Fuerte
(mezcla 55/85/100 %). Límites de seguridad: lado ≤ 16 384 px y ≤ 64 MP. `convert_to` (png/jpg) = solo cambiar de formato
(sin IA, mismo tamaño, calidad 92, sufijo `_convertido`). Salida junto al original (`_remaster`); si el origen es una
subida/pegado, en `Imágenes\Reescaller`; nunca se pisa un archivo existente (`nombre (2).ext`).

### 16.5 API local
| Ruta | Uso |
|---|---|
| `GET /api/ping`, `/api/focus`, `POST /api/quit` | Ciclo de vida |
| `GET/POST /api/config`, `POST /api/rescan` | Configuración y re-detección de modelos |
| `POST /api/restart`, `GET /lang/<código>.json` | Reinicio de la app y diccionarios de idioma |
| `GET /api/doctor`, `/api/hardware[?refresh=1]` | Diagnóstico y hardware (`refresh=1` vuelve a detectar las GPU; la caché ya no es permanente) |
| `GET /api/pick?kind=image|folder|video|model|exe`, `/api/reveal`, `/api/preview`, `POST /api/upload` | Archivos |
| `POST /api/remaster` (`dry_run`, `convert_to`), `/api/upscale`, `/api/batch`, `/api/video`, `/api/convert`, `/api/trim`, `/api/face` | Trabajos |
| `GET /api/logs?since=` | Progreso real (la UI lee «Progreso n/m» y el % del motor) |

### 16.6 Compatibilidad de hardware y detección
| Mejora | Detalle |
|---|---|
| Sonda del motor | El propio `realesrgan-ncnn-vulkan.exe` enumera las GPU con el índice exacto de `-g`; no depende de `vulkaninfo`. Máx. 20 s por si el driver se cuelga |
| GPU integrada vs dedicada | Tipo Vulkan si existe; si no, heurística por nombre (UHD/Iris/«Radeon Graphics»/780M…). `llvmpipe`, WARP y SwiftShader se descartan |
| VRAM real | Registro (`HardwareInformation.qwMemorySize`) en vez de `AdapterRAM` (tope 4 GB): la RX 6600 pasa de «4 GB o más» a 8 GB |
| `-g` explícito | Solo con índices de enumeración Vulkan real |
| Poca VRAM / fallos | Reintento con teselas de 100 px → GPU automática con teselas de 32 px → continúa sin IA (avisa en la UI) |
| Sin GPU / Vulkan | Modo CPU (Lanczos de alta calidad) con aviso visible |
| Runtime VC++ | El motor necesita `VCOMP140.DLL`; se incluye junto al motor y `doctor` lo comprueba |
| Procesos hijo | `CREATE_NO_WINDOW` (sin ventanas de consola) |
| **Búsqueda de modelos** | `autodetectar` mira junto a la app, **carpetas superiores** (hasta 3 niveles, p. ej. el proyecto que contiene `dist\`) y `%LOCALAPPDATA%\Reescaller\models`; incluye `venv\Scripts\python.exe`. Desde la UI se pueden fijar rutas a mano |
| Puente de rostros | `restore_face.py` va **embebido** en el ejecutable y se escribe en `%LOCALAPPDATA%\Reescaller\bridge`; `doctor` lo usa |
| Opcionales | GFPGAN / RestoreFormer++ / Python se muestran como «Opcional», no como «FALLA» |

Verificado en este equipo (Intel i5-9600KF + AMD RX 6600): detección, `-g 0`, remaster con IA, fallback CPU, motor
averiado, GFPGAN desde la app (9 s). **No probado en NVIDIA / Intel Arc / portátiles híbridos**: cubiertos solo por tests
unitarios de parseo; usar «Informe» en un equipo real para validarlos.

### 16.7 Empaquetado y pruebas
- `crear_exe.bat` (o `scripts\build_dist.ps1`) compila y monta `dist\Reescaller\` (`Reescaller.exe`, `reescaller-cli.exe`,
  `WebView2Loader.dll`, `models\` con motor, FFmpeg, `vcomp140.dll` y modelos ncnn) + ZIP portable (~81 MB).
  Parámetros: `-SkipBuild`, `-TargetDir`, `-DistRoot`. Si `Reescaller.exe` está abierto, avisa **antes** de borrar nada;
  vacía el contenido de `dist\Reescaller` en vez de borrar la carpeta (puede estar abierta en el Explorador).
- Instalador: `installer\Reescaller.iss` (Inno Setup, por usuario, sin administrador); `build_dist.ps1` lo compila si
  encuentra `ISCC.exe`.
- Compilación: `rust\build.bat` desde PowerShell; si el usuario tiene un `reescaller-debug.exe` abierto en
  `rust\target\release`, compilar con `CARGO_TARGET_DIR=rust\target-dist` (lo que usa el script).
- Pruebas: 26 tests unitarios (`cargo test -p reescaller-core`: hardware, parseo de sonda, remaster, imagen) y
  `reescaller-selftest` con 29 comprobaciones (entorno, reescalado, peso, lote, video, hardware, modo fácil, fallback sin
  GPU, motor averiado). La UI se verifica con Edge headless (`--screenshot`/`--dump-dom`); `REESCALLER_UI=<index.html>`
  recarga la interfaz sin recompilar.

### 16.9 Motor de rostros nativo (ONNX) — versión de prueba
Rostros sin Python ni torch: `reescaller-core::face_onnx` (ONNX Runtime con DirectML, `ort` en modo `load-dynamic`) +
`face::restore` (selección de motor). Config `face_backend` = `auto` | `onnx` | `python` (por defecto `auto`);
`REESCALLER_DML_DEVICE=N` elige el adaptador DirectML. Archivos en `models\onnx` (`onnxruntime.dll`, `DirectML.dll`,
`retinaface_r50.onnx`, `parsenet.onnx`, `gfpgan_v14.onnx`, `restoreformer_pp.onnx`; ~816 MB + 38 MB de DLL), generados con
`experimental/onnx-faces/export_onnx.py`. `scripts\build_dist.ps1 -WithOnnx [-NoZip] [-DistRoot ...]` los incluye.
Robustez verificada (grises, alfa, 16 bits, EXIF, borde, caras pequeñas, hasta 9000 px, respaldo GPU→CPU) y licencias:
ver `experimental/onnx-faces/README.md`. Sigue sin probarse en NVIDIA/Intel; GFPGAN trae componentes de licencia
no comercial (revisar antes de un uso comercial).

### 16.10 Correcciones tras las pruebas de usuario
- **Aviso de seguridad de Windows al remasterizar**: `realesrgan-ncnn-vulkan.exe` y `ffmpeg.exe` llevaban la marca «descargado
  de Internet» (`Zone.Identifier`, ZoneId=3), lo que hace que Windows muestre «Abrir archivo - Advertencia de seguridad» cada
  vez que la app los lanza. Ahora `build_dist.ps1` los desbloquea (`Unblock-File`) y la app elimina esa marca de sus `.exe`/`.dll`
  al arrancar (`process::unblock_downloaded`), también si el usuario descomprime un ZIP descargado.
- **Motor y FFmpeg propios primero**: `autodetectar` da prioridad a los que van junto a la app sobre una ruta guardada de otra
  instalación (antes una carpeta de pruebas anterior podía "secuestrar" la configuración).
- **Sin Python/torch en la UI**: se quitaron del diagnóstico, de «Modelos y extras» y de la vista de rostros; el motor se
  presenta como «Restauración de rostros». El puente de Python sigue existiendo solo como respaldo interno (`face_backend`).
- **Pegar imágenes recortadas** (Ctrl+V o botón «Pegar») en Restaurar rostros e Imagen y carpeta; arrastrar y soltar usa la
  misma lógica según la vista activa.
- **Convertir video**: ahora recodifica cuando la copia directa no es posible (WebM → VP9 + Opus, MP4/MOV/MKV → H.264 + AAC,
  AVI → MPEG-4 + MP3). Los errores de FFmpeg se resumen a sus últimas líneas útiles.
- **Interfaz**: tarjeta del plan sin anidar en Imagen y carpeta, imágenes centradas en las comparativas, textos largos con
  ajuste de línea y avisos truncados. La pestaña «Desarrollo» (editor de `config.json`) queda oculta; se muestra con
  Ctrl+Mayús+D.

### 16.11 Vista compartida, temas de consola y refrescar dispositivos
- **Vista compartida**: la tarjeta de entrada (`#card-input`) y la vista previa (`#stage-card`) se montan en «slots» y se
  mueven entre Remasterizar y «Una imagen» (`mountShared`), por lo que comparten estado (`S.input`, `S.result`) y la imagen
  elegida o pegada no se pierde al cambiar de vista. `runUpscale` pinta el resultado con el mismo `renderImageResult`.
  `/api/upscale` sin salida explícita usa la misma regla que el modo fácil (junto al original o `Imágenes\Reescaller` si viene
  de una subida/pegado).
- **Temas nuevos** (claro y oscuro cada uno, nombres propios inspirados en consolas/juegos): **Esmeralda** (verdes tipo Xbox),
  **Zafiro** (azules con acento rosa, tipo PlayStation), **Magma** (rojo/naranja, tipo Doom Eternal) y **Distrito Neón**
  (amarillo/magenta/cian, tipo Cyberpunk 2077). Se definen como bloques `html[data-palette="..."]` (+ `[data-theme="light"]`)
  generados con el mismo conjunto de variables que Miami; el tema se guarda en `config.theme` (p. ej. `distrito-light`).
- **Refrescar dispositivo de cálculo**: botón en esa tarjeta; `hardware::refresh()` sustituye la caché de detección y la UI
  actualiza hardware, selector de GPU y tabla de estado.
- La pestaña «Desarrollo» sigue oculta (Ctrl+Mayús+D).

### 16.12 Temas animados, idiomas e icono por tema
- **Fondo animado** (`Ajustes → Temas → Efectos e icono → Fondo animado`, activado por defecto): con `html[data-anim="on"]` las cinco
  manchas de color del fondo se desplazan, cambian de tamaño y de opacidad (`drift1/2/3`) y una capa cónica con los tres colores del
  tema gira despacio (`bgspin`, 46 s). Con «off» el fondo queda estático. Se guarda en `config.animated_theme` y en
  `localStorage` (`reescaller-anim`); respeta `prefers-reduced-motion`. Verificado con estilos calculados: la transformación cambia
  con el tiempo en «on» y es `none` en «off».
- **Idiomas**: Español (texto base), English, 中文, 日本語, Русский, 한국어, Português, Français, Deutsch e Italiano, más «Automático»
  (idioma del sistema, `navigator.language`; si no está soportado usa inglés). Selector en `Ajustes → Idioma`; se aplica al instante,
  sin recargar, y se guarda en `config.language` + `localStorage` (`reescaller-lang`).
  - Motor (`index.html`, sección «Idiomas»): los diccionarios son `assets/lang/<código>.json`, con la **cadena en español como clave**
    (los valores variables usan `{0}`, `{1}`…) y se sirven desde `GET /lang/<código>.json` (embebidos con `include_str!`; con
    `REESCALLER_UI` se leen del disco sin recompilar). Un recorrido del DOM traduce nodos de texto y los atributos
    `title/placeholder/aria-label/alt` y un `MutationObserver` traduce todo lo que la interfaz inserta después (se conserva el
    original para poder volver a otro idioma). La guía animada traduce párrafos completos con `t()`. `translate="no"` excluye
    zonas (nombres de idioma).
  - **Para añadir un idioma**: crear `assets/lang/xx.json`, añadirlo a `LANGS` (index.html) y a `lang_response` (server.rs).
  - Los mensajes de error técnicos que vienen del núcleo (Rust) y los nombres de hardware siguen en español/original.
- **Icono de Windows por tema (se aplica al reiniciar)**: `icon.rs` dibuja en tiempo de ejecución el logo (píxel sólido + esquinas) sobre un
  degradado con los dos colores del tema guardado en `config.theme` (`palette_colors`, variantes clara/oscura con los mismos `--tint` /
  `--tint-2` de la interfaz; supermuestreo 3×3, 128 px). Se usa al crear la ventana (`with_window_icon` + `with_taskbar_icon`, así el
  botón de la barra de tareas también lo recibe). **No cambia en vivo**: al cambiar de tema la interfaz muestra el aviso «Reinicia para
  ver el icono del nuevo tema» (`S.iconTheme` frente al tema actual) hasta que se reinicia. El icono incrustado en el `.exe`/acceso
  directo **no puede cambiar** (es un recurso del archivo); con la app anclada Windows puede conservar el del acceso directo.
- **Reiniciar aplicación** (`POST /api/restart`, botón en Temas, con diálogo propio `askConfirm` en lugar de `confirm()`): la interfaz
  guarda tema/idioma/animación y llama al endpoint; `window::restart_self()` abre otra instancia del mismo `.exe` con los mismos
  argumentos más `--restarted` (desacoplada del proceso actual) y solo responde `ok` si se pudo lanzar; el proceso actual sale 0,4 s
  después. La instancia nueva espera hasta 20 s a que el puerto se libere y reutiliza el mismo puerto (la interfaz guarda sus
  preferencias por origen). El log (`%LOCALAPPDATA%\Reescaller\logspp.log`) registra «Reinicio solicitado/fallido».
- **Error corregido (v. prueba-onnx-5/6)**: al calcular el icono inicial se dejaba el *mutex* de la configuración bloqueado mientras vivía
  la ventana (el guard temporal duraba toda la llamada a `native::run`), de modo que `GET/POST /api/config`, el diagnóstico, etc. se
  quedaban colgados y «Reiniciar» parecía no hacer nada. Ahora el tema se lee antes, en una sentencia aparte.
- **Verificado** con el `.exe` real (puerto de prueba): arranca con el icono del tema guardado (ventana y barra de tareas, `WM_GETICON`
  grande y pequeño), guardar otro tema **no** cambia el icono, y tras `POST /api/restart` el proceso anterior termina y el nuevo abre
  con el icono del nuevo tema (zafiro-claro → magma-oscuro). Capturas de la interfaz en ja, zh, ru, ko y de sin textos cortados.
  **No verificado**: el icono con la app anclada a la barra de tareas; que las traducciones (hechas con IA, no por nativos) suenen
  naturales en todos los casos.
- **Modo simple con todas las funciones** (antes solo imagen): pestañas **Imagen · Video · Rostros** en la cabecera (`mSetTool`). Cada una
  cambia el selector de acción, el de tamaño y el botón principal, y mantiene el estilo «chat» de una sola tarjeta:
  - **Imagen**: igual que antes (Auto/Foto/Anime/Web ligera, convertir a PNG/JPG, carpeta, tamaño final).
  - **Video**: *Mejorar* (IA, tamaño ×0.5–×2, salida `_mejorado.mp4`), *Convertir* a MP4/WebM/MKV/MOV/AVI y *Recortar* (inicio + duración
    en la propia barra). Elegir archivo con el selector nativo; la zona central muestra el nombre del archivo.
  - **Rostros**: modelo *Rápido* (GFPGAN) o *Alta calidad* (RestoreFormer++) y comparador antes/después.
  - Las llamadas son las mismas de la vista completa (`/api/video`, `/api/convert`, `/api/trim`, `/api/face`); «Modo avanzado» lleva el
    archivo a la vista que corresponde (Video o Restaurar rostros). Pegar o arrastrar una imagen desde la pestaña Video cambia a Imagen.
  - Renombrado: el botón «Modo mini» pasa a **«Modo simple»** y «Opciones avanzadas» (en la tarjeta simple) a **«Modo avanzado»**, en la
    interfaz, la guía y los 9 idiomas.
  - **Tamaños estables**: la barra inferior usa columnas fijas (`1.25fr / 1.35fr / 172px`) y el botón de elegir archivo tiene ancho mínimo
    fijo, de modo que al cambiar de pestaña o de acción ninguna caja cambia de tamaño (medido en es y ja: 223×55, 241×55 y 167×55 px en
    todas las pestañas/acciones). El botón de carpeta se atenúa (`.off`) en vez de desaparecer, para no mover el nombre del archivo.
  - **Animaciones de cambio**: indicador deslizante de las pestañas (`#m-ind`, recolocado con `ResizeObserver` y al cambiar de idioma),
    entrada escalonada del contenido central (`mSwapIn`), «pop» de las cajas de la barra y del botón principal al cambiar de
    herramienta/acción, y aparición suave de las casillas de recorte y del selector de tamaño. Respetan `prefers-reduced-motion`.
  - Verificado con la API simulada (cuerpos de petición correctos para mejorar/convertir/recortar/rostros) y capturas en es/en/ja; no
    probado procesando un video real desde esta pestaña.
- **Lenguaje normal en vez de nombres técnicos** (ronda 16.13):
  - **Modo simple · Imagen**: la acción pasa a **niveles** — *Auto*, *Nivel 1 (Suave)*, *Nivel 2 (Normal)*, *Nivel 3 (Fuerte)* y *Web ligera*
    (`mImageOptions()` los traduce a preset + intensidad) — más convertir a PNG/JPG. El tamaño incorpora **calidad**: *Tamaño auto*,
    *Mismo tamaño*, *Estándar ×2*, *Alta ×3*, *Máxima ×4*. Foto/Anime siguen en la vista Remasterizar.
  - **Modo simple · Video**: además de mejorar, convertir (MP4/WebM/MKV/MOV/AVI) y recortar, **conversiones extra**: *a GIF* (primeros 20 s,
    480 px, 12 fps, paleta optimizada) y *a MP3* (solo audio). Implementado en `video::convert_video` (ffmpeg `palettegen/paletteuse` y
    `libmp3lame`); también en el formato de salida de la vista Video. Probado con ffmpeg real: GIF 480×270 de 4 s y MP3 de 4 s válidos.
  - **Menús con descripción**: las opciones pueden llevar `data-desc` y el menú personalizado la muestra bajo el nombre (traducida).
  - **Modo avanzado**: modelos como *Fotos y realismo / Anime e ilustraciones / Rápido* (y rostros *Rápido / Alta calidad*), filtros como
    *Máxima calidad / Nítido / Suave / Píxeles*, formato PNG/JPG con descripción. El plan («Modelo «…»», Filtro, Modo, Dispositivo) usa los
    mismos nombres (`MODEL_LABELS`, `FILTER_LABELS`…); los valores internos (`realesrgan-x4plus`, `lanczos3`…) no cambian.
  - **Botones redundantes fuera**: en Remasterizar / Imagen y carpeta (tarjeta de entrada compartida) queda solo «Elegir imagen» (pegar es
    Ctrl+V —también tras copiar o cortar—, arrastrar sigue funcionando y las carpetas van en «Carpeta completa»); en Restaurar rostros se
    quitan «Subir» y «Pegar» (queda el selector nativo + Ctrl+V). El modo simple conserva su botón de carpeta.
  - Los patrones de traducción ahora también traducen los valores capturados (p. ej. el nombre amable del modelo dentro de la frase del plan).
- **Pegar imágenes de forma fiable y botón X** (ronda 16.14):
  - Problema: las capturas del Recortes de Windows a veces no llegaban como archivo en el evento `paste` del navegador, así que «a veces
    cargaba y otras no». Ahora el evento también mira `clipboardData.items` y, si no hay imagen, la app **lee el portapapeles del sistema**
    (`GET /api/clipboard`, módulo `clipboard.rs` con `clipboard-win`): archivos de imagen copiados/cortados en el Explorador (se usan
    en su sitio), PNG registrado (Recortes) y mapa de bits clásico (se guardan como PNG en `Imágenes de subida`). Si el navegador
    no llega a generar `paste`, Ctrl+V dispara la misma lectura a los 250 ms. Además, al recuperar el foco la ventana devuelve el teclado
    a la página (`webview.focus()`), para que Ctrl+V funcione tras usar la barra de título.
  - Probado con el portapapeles real de Windows (STA): bitmap, PNG, archivo copiado y portapapeles sin imagen (mensaje claro).
    Flujo de interfaz probado con API simulada en Remasterizar, Restaurar rostros y modo simple (cargar con Ctrl+V y quitar con la X).
  - **X para quitar la imagen**: en la vista completa, en la cabecera de «Vista» (`#stage-x`, también tras procesar); en Restaurar
    rostros, sobre la imagen original (`#face-x`, limpia ruta, vistas previas y resultado); en el modo simple ya existía en la zona central.
- **Botón principal único, X que funciona y arranque sin pantalla en blanco** (ronda 16.15):
  - **«Comenzar»** es ahora el texto de todos los botones de ejecutar (modo simple, Remasterizar, Imagen y carpeta, Video —mejorar,
    convertir y recortar— y Restaurar rostros); el contexto queda en el título del botón (tooltip) y en la tarjeta donde está.
  - **X del modo simple**: no funcionaba porque el comparador capturaba el puntero y se tragaba el clic (`setPointerCapture`); ahora el
    comparador ignora los botones y la X ya no está sobre la imagen: ocupa el lugar del botón de carpeta (se vuelve roja con «Quitar»
    cuando hay archivo cargado y vuelve a ser carpeta al quitarlo). En la vista completa y en Restaurar rostros se mantienen las X anteriores.
  - **Arranque**: la ventana nace oculta y se muestra cuando la interfaz pinta su pantalla de arranque (logo con barra animada,
    `#splash`; respaldo: se muestra a los 4 s); el WebView2 usa fondo oscuro (sin destello blanco). La pantalla de arranque se quita
    con un desvanecimiento (mínimo 700 ms). **Primera vez**: se abre sola la guía animada «Cómo se usa» (`localStorage reescaller-welcomed`).
    La zona vacía del modo simple late con un borde naranja discontinuo para invitar a soltar o elegir una imagen.
- **Temas nuevos y compatibilidad de hardware** (ronda 16.16):
  - **Carmesí** (nuevo): negro puro (OLED) con rojo carmesí, claro y oscuro. **Miami claro** rediseñado: turquesa y rosa (Miami Vice de día) sobre
    blanco frío en lugar del degradado melocotón/violeta. **Distrito Neón** rehecho al estilo HUD ciencia ficción: rojo neón y cian sobre negro, rejilla
    de fondo, esquinas de marco cian/rojo en las tarjetas y radios pequeños (claro: rojo/verde azulado sobre gris cálido). Las franjas de color de la
    lista de temas cambian según claro/oscuro; el icono de ventana usa los nuevos colores (`icon.rs`).
  - **Compatibilidad de modelos** (`reescaller-core/src/compat.rs`, `GET /api/compat`, `POST /api/compat/test`, tarjeta en Ajustes → Equipo y GPU):
    clasifica cada GPU por nombre y generación (NVIDIA: GTX 600 en adelante = Kepler…Blackwell; AMD: RX 400/500 = Polaris en adelante; anteriores = «por
    debajo del mínimo») y por memoria de video, y da un veredicto por modelo —**Compatible / Limitado / No compatible**— con los motivos (sin Vulkan, GPU
    antigua, poca VRAM, GPU integrada, uso de CPU). Modelos: Fotos y realismo, Anime e ilustraciones, Rápido (motor Vulkan) y Rostros Rápido / Alta
    calidad (DirectML o CPU). «Probar ahora» **ejecuta de verdad** cada modelo del motor sobre una imagen de 48×48 en la GPU elegida (límite de 60 s)
    y comprueba el tamaño de salida. Bajo los selectores de modelo aparece un aviso si el elegido es limitado o no compatible. Probado en Intel + RX 6600:
    todo «Compatible» y las 3 pruebas reales superadas en 6 s; la clasificación por nombre tiene pruebas unitarias (GTX 650/680/1060/1660, RTX 4070,
    GTX 480/560, GT 710, RX 470/580/6600/7900, R9 290, HD 7970, Vega). **No probado** en hardware NVIDIA/AMD antiguo real.
  - Auditoría de seguridad/bugs: ver `docs/AUDITORIA.md`.
- **Ajustes de temas y botones de ventana** (ronda 16.17):
  - **Distrito Neón** sin cuadrícula, sin marcos de esquina y con las mismas formas (radios, botones) que el resto de temas; se mantienen sus colores.
  - **Movimiento del fondo algo más agresivo** en todos los temas: ciclos 30-35 % más rápidos (12-20 s), recorridos más largos, más variación de opacidad
    y la capa cónica gira en 30 s en vez de 46 s.
  - **Temas claros más vivos**: degradados con más color en Clásico (lavanda→aqua), Miami, Carmesí, Esmeralda, Zafiro, Magma y Distrito, y fondo animado más visible
    en claro (opacidad .48).
  - **Tema nuevo «Unidad 01»** (id `eva`): armadura violeta con detalles verde neón y toques naranja, claro y oscuro (inspirado en los mechas de anime; el nombre no
    usa marcas). El icono de la ventana también lo usa.
  - **Botones de ventana**: iconos SVG propios (minimizar, maximizar/restaurar —cambia según el estado real de la ventana con `syncMaxed()`—, cerrar), 40×32 px,
    con estados hover/pulsado y cierre en rojo. Probado con una ventana simulada: los clics envían `min`, `max` y `close` por IPC y el icono de restaurar aparece al maximizar.
  - **Error de arranque corregido**: la ventana nace oculta y el primer `ShowWindow` hereda el estado pedido por quien lanza el proceso; si el lanzador pedía «minimizada»
    (acceso directo configurado así, ciertos lanzadores) la app aparecía minimizada. Ahora se llama también a `set_minimized(false)` al mostrarla.
- **Tema personalizado, checks con el color del tema e interruptores nuevos** (ronda 16.18):
  - **«Ok» con el color de la interfaz**: `--ok` ahora es `var(--tint-2)` en todos los temas (chips «Compatible», estado «OK», marcas ✓, avisos de éxito), con
    ajustes en Carmesí (blanco) y Magma (amarillo) para que no se confundan con el color de error.
  - **Personalizado** (Ajustes → Temas, última tarjeta): se parte de un **tema base** (los 8 temas) y se ajustan **color principal, color secundario y fondo**
    por modo claro/oscuro, con vista previa y «Restablecer». A partir de esos tres colores se derivan degradado de fondo, manchas animadas, texto, tinta de
    rellenos, menús, avisos y código (`customVars`), calculando el contraste del texto según la luminosidad del fondo. Se aplica con propiedades CSS en línea
    (`applyCustomVars`) y se guarda en `localStorage` (`reescaller-custom`) y en `config.custom_theme` (JSON); `config.theme` queda como `custom-dark`/`custom-light`.
    El icono de ventana usa los colores personalizados al arrancar (`icon::palette_colors_cfg`, con prueba unitaria).
  - **Interruptores** rediseñados: 52×30 px, pista de color sólido del tema con una marca ✓ que aparece al activar, pomo que se desliza con rebote, borde y pomo
    apagados en gris, anillo de foco accesible y leve efecto al pulsar; las casillas usan el color del tema (`accent-color`).
- **Selector de color propio y acentos legibles** (ronda 16.19): el diálogo nativo de Chromium se sustituye por un selector con el estilo de la app
  (`openColorPop`): **paleta por defecto** (32 acentos vivos para principal/secundario y 16 fondos oscuros o claros según el modo), **cuadro de saturación/brillo**,
  **barra de tono**, vista previa y **campo hex** validado; se aplica en vivo, se cierra con Esc, al hacer clic fuera o al desplazar la página, y se coloca arriba o
  abajo según el espacio. Para que ningún color elegido deje texto ilegible, el tema personalizado calcula `--accent-text` y `--accent-icon` acercando el color a
  blanco/negro hasta alcanzar contraste 4.6:1 (texto) / 3.2:1 (iconos) con el fondo; los textos e iconos de acento del resto de la interfaz usan esas variables (en los
  temas predefinidas valen `--tint-2` y `--tint`). El aviso «Reinicia para ver el icono del nuevo tema» ahora es una etiqueta con fondo y borde del color de acento.
  Probado: abrir el selector, elegir de la paleta, arrastrar en el cuadro y en el tono, y escribir un hex actualizan el color guardado.
- **Botones con animación, ajustes rediseñados y salida sencilla** (ronda 16.20):
  - **Botones que trabajan** (Refrescar, Probar ahora, Comprobar, Volver a detectar, Buscar de nuevo, Informe): `withBtn()` les pone el estado `loading`
    (icono sustituido por un spinner, barrita de progreso que barre el borde inferior, bloqueo de clics) un mínimo de 0,6 s para que se note.
  - **Panel «Ajustes»** (antes «Plan de reescalado»): opciones por botones y **número personalizado** que se entiende —*Tamaño final* (Mismo, ×1.5, ×2, ×3, ×4 o un número),
    *Intensidad de la mejora* (Suave 55 %, Normal 85 %, Fuerte 100 % o un %), *Nitidez* (Nada/Suave/Media/Fuerte o 0-100), *Modelo*, *Archivo de salida* (PNG/JPG) y
    *Calidad del archivo* (Alta 95, Media 85, Ligera 70 o 1-100)—; lo técnico (escala del motor, cuándo reescalar, peso objetivo en KB, filtro, usar motor de IA)
    queda en un desplegable «Ajustes avanzados». Los botones y las casillas numéricas gobiernan los mismos controles de siempre (`PLAN_CHIPS`/`syncPlanUI`), así
    que el plan enviado al servidor no cambia. Probado: cada opción actualiza el plan (×3, 70 %, nitidez 20, calidad 95, JPG).
  - **Carpeta**: se elige una sola carpeta y las imágenes mejoradas se guardan **dentro de ella, en «Mejoradas»** (se muestra la ruta final); la carpeta de salida
    propia y los trabajos simultáneos pasan a «Más opciones». `/api/batch` acepta `output` vacío (por defecto `<carpeta>\Mejoradas`).
  - **Nombres de salida**: las imágenes se guardan como `<nombre>_mejorado.<ext>` (antes `_remaster`) y los videos como `<nombre>_mejorado.mp4`; convertir
    mantiene `_convertido` y recortar `_recortado`. En las tarjetas de video la ruta de salida queda oculta (texto «Se guarda junto al video con el sufijo …» + «Cambiar»).
  - **Cómo se usa**: sin botón de pausa/reproducir; avanza sola y se detiene al navegar a mano.
- **Pulido de la animación de botones y de los campos numéricos** (ronda 16.21): el spinner ya no se superpone al texto (el icono se oculta y el spinner va delante, con espacio) y
  la barra de progreso recorre todo el borde inferior; los campos numéricos pierden las flechas nativas y usan botones − / + propios (paso y límites del campo), con
  unidades alineadas.
- **Endurecimiento de seguridad, robustez e instalador (v0.3.0)** (ronda 16.22; hallazgos de `docs/AUDITORIA.md`):
  - **A3 token de sesión**: cada ejecución genera un token aleatorio de 128 bits que el servidor inyecta en el HTML (`<meta name="reescaller-token">`) y exige en
    `X-Reescaller` en toda la API (salvo `ping`/`focus`, que usa la segunda ejecución de la app). También se rechazan peticiones con `Origin` ajeno o
    `Sec-Fetch-Site` distinto de `same-origin`/`none`, cuerpos de más de 160 MB, y el HTML lleva CSP (`default-src 'self'`, `frame-ancestors 'none'`) y `nosniff`.
  - **A1**: «Abrir» (`reveal?open=1`) solo abre imágenes, video y audio (nunca ejecutables ni scripts); las subidas solo aceptan imágenes, con nombre saneado y
    comprobación de cabecera (`image::guess_format`); se rechazan rutas de red (UNC) en vista previa y «Mostrar»; `preview` limita `max` a 4096.
  - **A2**: `POST /api/config` solo acepta `realesrgan…exe`, `ffmpeg.exe` y `python*.exe` como programas.
  - **A4**: «Mostrar en carpeta» ya no construye código con la ruta (atributo `data-reveal` escapado + un único escuchador) y `esc()` escapa comillas y `&`.
  - **A5**: la app se mete en un *Job Object* con `KILL_ON_JOB_CLOSE` (`window::kill_children_on_exit`): al cerrar, Windows termina `ffmpeg`/`realesrgan` hijos (se permite
    el *breakaway* para que el reinicio siga funcionando); cerrar con un trabajo en curso pide confirmación.
  - **A6**: el video usa una carpeta temporal única (`reescaller_tmp_<pid>_<ms>`) que se borra siempre (guard `Drop`), sin depender del nombre del video.
  - **M1**: las imágenes subidas o pegadas de más de 24 h se borran al arrancar. **M3**: el reescalado aplica la orientación EXIF (fotos de móvil). **M6**: un resultado «sin
    Vulkan» se vuelve a comprobar a los 30 s (la sonda puede fallar una vez) y el bloqueo de la caché ya no se mantiene durante la detección.
  - **Instalador** (`installer/Reescaller.iss`, Inno Setup 6): `dist\Reescaller-0.3.0-Setup.exe`; por usuario (sin administrador), **Windows 10 1809+ y Windows 11 de 64 bits**,
    componentes (principal + «Restauración de rostros» ~800 MB opcional), instala **Microsoft Edge WebView2 Runtime** (bootstrapper firmado por Microsoft, verificado en
    `build_dist.ps1`) si falta, avisa si no hay driver con Vulkan y al desinstalar borra caché de la ventana y subidas. `Cargo.lock` ya se versiona (compilaciones reproducibles).
- **Rostros: «Retrato completo»** (ronda 16.23): GFPGAN y RestoreFormer++ solo reconstruyen la cara; el pelo, la ropa y el fondo quedaban como en la foto original. `/api/face` acepta
  `scope`: `portrait` (por defecto) amplía primero **toda la imagen** con el motor general (Real-ESRGAN ×2) y después restaura los rostros encima (`face::restore_scaled` con
  ampliación 1 para no ampliar dos veces); `face` mantiene el comportamiento anterior. Si la imagen supera 6 MP, no hay motor o no hay Vulkan, se avisa en el registro y se
  hace solo la parte de rostros. En la interfaz: selector «Alcance» (Retrato completo / Solo rostro) en Restaurar rostros y cuatro opciones en el modo simple (Retrato rápido/
  calidad, Rostro rápido/calidad). Probado con una foto real de 700×464: el pelo y el lazo salen nítidos frente al resultado «solo rostro» (8,4 s frente a 5,1 s).
  Modelos candidatos para ir más allá (no integrados): CodeFormer (también solo rostro), GPEN-BFR (recorte más amplio, algo de pelo) y modelos de difusión (SUPIR, DiffBIR:
  muy buenos con pelo y piel pero necesitan GPUs con 12 GB o más y tardan minutos).
- **Icono del `.exe` corregido** (ronda 16.24): con la cadena GNU el recurso (icono + versión) quedaba en una biblioteca estática que el enlazador descartaba, así que el ejecutable
  salía sin icono ni datos de versión. `build.rs` ahora pasa `resource.o` directamente a los ejecutables (`cargo:rustc-link-arg-bins`). Verificado: el `.exe` muestra el logo y
  `ProductName`/`FileVersion` (0.3.0); el instalador usa el mismo icono.
- **Rostros: más calidad y un instalador con el estilo de la app** (ronda 16.25):
  - **Ampliación automática ×4** en el retrato completo cuando la imagen es pequeña (lado largo < 640 px; ×2 en el resto), con tope de píxeles.
  - **Fotos oscuras** (`light`: `auto` por defecto / `on` / `off`): `image_ops::lift_dark` aclara con gamma hacia ~105/255 y suaviza un poco el ruido antes de restaurar; en `auto`
    solo actúa si la luminosidad media es < 75. **Fuerza de la restauración** (`strength` 0-100, `face_onnx::set_strength`): mezcla el rostro restaurado con el recorte original antes de pegarlo.
  - **Comparador con deslizador y zoom** en Restaurar rostros (rueda para acercar hasta ×8, arrastrar para mover, divisor fijo en pantalla, botones −/+/Ajustar), con vistas previas de 2400 px.
  - Probado con una foto real de 248×329 y mucha oscuridad: salida de 992×1316, brillo medio 42 → 83 y nitidez ~15× mayor que un agrandado simple; se ven mechones de pelo y detalle del ojo.
  - **Instalador con el estilo de la app** (`installer/Reescaller.iss` + `installer/make_assets.py`): tema oscuro (`WizardStyle=modern dark`), página de bienvenida y final con un panel lateral
    en «cristal» (manchas violeta/cian/rosa, logo y lema) en 4 tamaños de DPI, icono de cabecera, textos propios en español e inglés y barra de color de la marca. `ISCC /DQUICK=1` compila una
    versión sin la app para revisar el aspecto en segundos. Las imágenes se regeneran con `python installer\make_assets.py`.
- **Instalador propio estilo Opera/AMD/Adobe** (ronda 16.26, crate `rust/crates/reescaller-setup`): una sola ventana sin marco (920×600, cristal oscuro con manchas de color que
  cambian de paleta), bienvenida con comparador antes/después animado, botón **«Aceptar e instalar»** y desplegable **«Opciones»** (carpeta, restauración de rostros, acceso directo,
  abrir al terminar). Al instalar muestra una **presentación de funciones** (imágenes, retratos, video, temas e idiomas, privacidad) que cambia los colores de toda la ventana, y abajo
  una **barra de progreso** con %, archivo actual y tiempo restante; al final pantalla de éxito (con aviso si falta Vulkan) y error con «Reintentar». Español/inglés según el sistema.
  - **Arquitectura**: `Reescaller-<ver>-Setup.exe` = *lanzador* (`reescaller-setup-launcher`, sin DLL) + ZIP al final con `ui/reescaller-setup.exe`, `ui/WebView2Loader.dll`, los archivos de
    la app y `redist/MicrosoftEdgeWebview2Setup.exe`. El lanzador extrae la interfaz a una carpeta temporal (con su DLL: con la cadena GNU la interfaz WebView2 no arranca sin
    `WebView2Loader.dll` al lado) y la ejecuta con `--payload=<instalador>`; la interfaz lee la app de ese ZIP. Instala por usuario en `%LOCALAPPDATA%\Programs\Reescaller`, crea accesos
    directos (inicio y escritorio), registra «Aplicaciones instaladas» (HKCU) y copia la interfaz como `Uninstall.exe`; si falta WebView2 lo instala antes de abrir la ventana.
  - **Desinstalador** (`Uninstall.exe`, ventana propia; opción de borrar ajustes): quita accesos, registro y archivos; una orden `cmd` aparte reintenta borrar la carpeta hasta que el
    desinstalador se cierra (el `.exe` y la DLL cargada no pueden borrarse mientras corren).
  - **Ritmo de la presentación** (ajuste tras la primera prueba): el disco copia ~1 GB en segundos y no daba tiempo a leer las funciones, y la app se abría sola al terminar. Ahora la barra mostrada
    va acompasada con la presentación (mínimo 33 s = 5 diapositivas × 6,5 s; `tick()` en `setup.html` usa `min(progreso real, tiempo/33 s)`) y la pantalla final solo aparece cuando la barra
    llega al 100 %; los puntos son clicables para volver a una diapositiva. **La app ya no se abre sola**: el usuario pulsa «Abrir Reescaller» (se quitó la opción «Abrir al terminar»).
    Al desinstalar y en modo silencioso no hay retraso. Verificado con una simulación en Edge sin ventana (final a ~33 s, sin ninguna orden `launch`).
  - **Modo silencioso** para scripts: `Setup.exe /S --dir=RUTA [--no-faces] [--no-desktop]`, `Uninstall.exe /S [--purge]`.
  - **Empaquetado**: `scripts\build_setup.ps1` (compila, verifica la firma de WebView2 y llama a `scripts\pack_setup.py`); `build_dist.ps1` lo ejecuta en el paso 5. El de Inno queda como
    alternativa: `dist\Reescaller-<ver>-Setup-Inno.exe`.
  - **Verificado** (sin pulsar la interfaz): instalación silenciosa completa (999 MB, modelos idénticos por hash, `Uninstall.exe`, registro y acceso), desinstalación silenciosa con código 0
    y carpeta, registro y acceso eliminados; captura de la ventana de bienvenida. **Pendiente de prueba manual**: recorrer la interfaz con clics (opciones, presentación, barra,
    éxito, desinstalar), WebView2 ausente en un Windows 10 limpio y SmartScreen (sin firma de código).
- **Limpieza y entrega** (ronda 16.27): `cargo clippy --all-targets` sin avisos en el código de producción (8 corregidos; queda 1 estilo en un test) y `cargo fmt --all` aplicado; 31 tests
  unitarios en verde. `README.md` reescrito para la app Rust (el Python queda como legado), `ROADMAP.md` nuevo, `.gitignore` con `installer/redist/`, y `scripts\make_source.ps1` crea
  `dist\Reescaller-<ver>-source.zip` (solo lo que Git versionaría, ~1 MB) y `dist\SHA256SUMS.txt`. Entregables de `dist\`: `Reescaller\` (portable), `-win64.zip` (835 MB), `-Setup.exe`
  (899 MB, instalador propio), `-Setup-Inno.exe` (815 MB) y `-source.zip`. Instalador final verificado en silencio (999 MB, versión 0.3.0, rostros incluidos, desinstalación limpia).
- Campos nuevos en `config.json`: `language` (`auto` por defecto) y `animated_theme` (`true`); ambos con valor por defecto para que
  las configuraciones antiguas sigan cargando.

### 16.8 Pendiente
- **Hoja de ruta de dependencias nativas e instalador de componentes** (migración de rostros a ONNX + asistente
  de primer arranque): ver [`docs/RUTA_MIGRACION_NATIVA.md`](docs/RUTA_MIGRACION_NATIVA.md). Es un documento de viabilidad
  con criterios de go / no-go. La prueba aislada de rostros con ONNX (`experimental/onnx-faces/`) ya demostró que es
  **viable**: pipeline completo en Rust sin Python/torch, 3.5 s frente a 36 s del puente en GPU (ver §0 del documento).
  Aún no está integrada en la app.
- Probar en NVIDIA, Intel Arc e iGPU Intel/AMD reales; firmar el ejecutable (SmartScreen avisará sin firma).
- Instalador: opcionalmente incluir `vc_redist.x64.exe`; firmar el instalador y el desinstalador.
- Revisar la licencia de la compilación de FFmpeg antes de redistribuir.
- Los rostros (GFPGAN/RestoreFormer++, ~614 MB + Python con torch) no viajan en el ZIP: siguen siendo un extra
  opcional; hacerlos 100 % Rust implica exportarlos a ONNX (`ort`).
- Recortar clip está desactivado hasta que haya edición de video; comprobar con clics reales los botones de ventana
  (– ▢ ✕) y el arrastre en el `.exe` (probados solo por captura/IPC).
