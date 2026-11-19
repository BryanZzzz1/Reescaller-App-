# Auditoría de Reescaller (seguridad, bugs y robustez)

Revisión de solo lectura hecha sobre el código del workspace `rust/` y la interfaz `index.html`. Cada punto se verificó leyendo las
líneas indicadas, salvo los marcados **[P]** (plausible, no comprobado en ejecución). No se ha cambiado nada por esta auditoría.
Rutas relativas a `rust/crates/`.

**Lo que ya está bien**: el servidor solo escucha en `127.0.0.1` y valida `Host`; sin cabeceras CORS el preflight falla (DNS rebinding y CSRF
desde webs externas quedan bloqueados); `reveal` no permite inyección de argumentos; los procesos se lanzan con argumentos separados, sin shell.

## Prioridad alta

| ID | Tema | Dónde | Problema y arreglo |
|---|---|---|---|
| A1 | Seguridad | `api.rs` 1022-1036, 466-499 | `/api/reveal?open=1` pasa el archivo a `explorer.exe`, que lo ejecuta (.exe, .bat, .lnk…); `/api/upload` acepta cualquier extensión. Subir `x.bat` + `reveal?open=1` = ejecución de código. **Arreglo**: en `open` solo imágenes/video y solo rutas de carpetas de la app; en upload, lista blanca de extensiones y comprobación de cabecera. |
| A2 | Seguridad | `api.rs` 619-632, `config.rs` 150-160 | `POST /api/config` guarda `realesrgan_exe`, `ffmpeg_exe` y `python_exe` sin validar; luego se ejecutan. **Arreglo**: aceptar solo rutas devueltas antes por `/api/pick` o validar el nombre del ejecutable y pedir confirmación nativa. |
| A3 | Seguridad | `server.rs` 53-66 | La única defensa de la API es una cabecera `X-Reescaller` fija y pública. Sin token de sesión, sin comprobar `Origin`/`Sec-Fetch-Site`, sin CSP. Cualquier proceso local controla toda la API. **Arreglo**: token aleatorio por arranque inyectado en el HTML y exigido en cada petición; rechazar `Origin` ajeno; añadir CSP. |
| A4 | XSS | `index.html` `revealBtn` | Construye un `onclick` con la ruta (`JSON.stringify(...).replace(/"/g,'&quot;')`) sin escapar `&`; un nombre de archivo hostil puede salir de la cadena y encadenar A1/A2. **Arreglo**: no construir `onclick` con datos; usar `addEventListener` y `dataset`. |
| A5 | Robustez | `server.rs`, `native.rs`, `process.rs` | Cerrar, `/api/quit` y `/api/restart` hacen `process::exit(0)` con trabajos en curso: `ffmpeg`/`realesrgan` quedan huérfanos y los temporales en disco. Sin Job Object, cancelación ni timeout. **Arreglo**: Job Object con `KILL_ON_JOB_CLOSE`, bandera de cancelación, timeouts y confirmación si hay trabajo. |
| A6 | Video | `video.rs` 80-108, 163-171, 187, 226 | El directorio de frames no se vacía antes de usarlo, los `?` salen sin limpiar (decenas de GB de JPEG), `remove_dir_all` puede borrar una carpeta del usuario con el mismo nombre, un `%` en el nombre rompe el patrón de ffmpeg y no se comprueba el espacio libre. **Arreglo**: carpeta temporal única propia, guard `Drop` de limpieza y comprobar espacio. |

## Prioridad media

| ID | Tema | Dónde | Problema y arreglo |
|---|---|---|---|
| M1 | DoS | `server.rs` 77-84, `api.rs` 467-499 | Cuerpo leído entero sin límite, un hilo por petición, `uploads` no se limpia nunca (ni al desinstalar). **Arreglo**: tope de `Content-Length`, limpiar `uploads` antiguos al arrancar, limitar hilos. |
| M2 | Robustez | `api.rs` 18-33, `image_ops.rs`, `upscale.rs` | `PlanDto` no se valida: un `target_scale` enorme revienta la memoria; `/api/preview?max=` sin tope. **Arreglo**: aplicar `limitar()`/clamp a todas las rutas. |
| M3 | Bug | `image_ops.rs` 15-20, 155-195 | El reescalado no aplica la orientación EXIF (fotos de móvil giradas), descarta el canal alfa y pasa 16 bits a 8. **Arreglo**: orientar al cargar, conservar RGBA. |
| M4 | Bug | `batch.rs` 119-125, `api.rs` | `a.png` y `a.jpg` se pisan entre workers; `unique_path` tiene carrera (comprobar y luego escribir). **Arreglo**: reservar el nombre con `create_new(true)`. |
| M5 | Datos | `api.rs` 210, 389-448 | La salida la elige el cliente y se sobrescribe sin aviso; `format` sin validar permite recorrido de rutas. **Arreglo**: lista blanca de formatos, `unique_path` también en video, rechazar salida == entrada. |
| M6 | Hardware | `hardware.rs` 238-254, 986-1002 | Si la sonda de Vulkan falla una vez (driver lento, GPU dormida en portátil híbrido), «sin GPU» queda cacheado toda la sesión y la IA se apaga en silencio; el `Mutex` se mantiene durante toda la detección. **Arreglo**: no cachear resultados con timeout, reintentar, soltar el lock. |
| M7 | Compatibilidad | `bin/reescaller-app.rs` | Sin WebView2/Edge/Chrome la app queda viva sin ventana. **Arreglo**: mensaje nativo con enlace al runtime. |
| M8 | Rendimiento | `remaster.rs`, `upscale.rs`, `api.rs` | `limitar` no cubre el intermedio ×4 del motor (16 MP → 256 MP), el PNG se decodifica dos veces y los workers de carpeta no miran la RAM. **Arreglo**: tope de píxeles de entrada, una sola decodificación, workers según RAM libre. |
| M9 | Seguridad | `window.rs`, `config.rs`, `face_onnx.rs` | Busca `realesrgan…exe`, `onnxruntime.dll` etc. en el directorio actual y hasta 6 carpetas superiores (binario plantado en Descargas). **Arreglo**: confiar solo en la carpeta del exe y `%LOCALAPPDATA%\Reescaller`; `LoadLibraryEx` con `LOAD_LIBRARY_SEARCH_*`. |
| M10 | Seguridad | `process.rs` 42-68 | En cada arranque se quita el `Zone.Identifier` de todos los .exe/.dll bajo la carpeta del proyecto. **Arreglo**: desbloquear solo binarios propios con hash conocido. |
| M11 | Robustez | `server.rs` 121-131 | Dos trabajos pesados simultáneos comparten temporales y logs. **Arreglo**: semáforo global de trabajos pesados. |
| M12 | Bug | `state.rs` 98-109, `process.rs` 127 | El id del log se asigna fuera del lock (líneas perdidas) y el progreso de ffmpeg (`\r`) llega de golpe al final. **Arreglo**: asignar dentro del lock y dividir también por `\r`. |
| M13 | Bug | `api.rs` 299-306, `video.rs` 213-218 | Video ignora `delete_temp`/`resize_filter`; `-c:a copy` a MP4 falla con Opus/PCM sin reintento; `-vsync 0` obsoleto. **Arreglo**: recodificar a AAC como respaldo y usar `-fps_mode passthrough`. |

## Prioridad baja

- **B1** (UI) Ventana fija de 840×620 lógicos: en portátiles de 1366×768 al 125 % los botones inferiores pueden quedar fuera de pantalla. Limitar al área de trabajo del monitor. [P]
- **B2** (Instalador) `UninstallDelete` borra `webview` pero la ventana usa `webview2`; no se borran las subidas; sin `AppMutex`; binarios sin firmar ni verificación por hash.
- **B3** (Scripts) `Where-Object` duplicado en `build_dist.ps1`; `WebView2Loader.dll` sin fijar versión; `build.bat` con ruta fija; `Cargo.lock` en `.gitignore` (compilaciones no reproducibles).
- **B4** (XSS defensivo) `esc()` no escapa comillas y se usa dentro de `title="…"`; hoy no explotable, pero frágil.
- **B5** (Dev) Con `REESCALLER_UI`, `/lang/<código>` se concatena a una ruta sin sanear (solo modo desarrollo).
- **B6** (Config) `config.json` se escribe sin atomicidad y, si es inválido, se vuelve a los valores por defecto en silencio; `app.log` sin rotación.
- **B7** (Exposición) `preview`/`browse`/`reveal` aceptan rutas UNC (fuga de hash NTLM, bloqueo del hilo). [P]
- **B8** (Cadena de suministro) El puente Python carga `.pth`/`.ckpt` (pickle) y antepone `models/` a `sys.path`. Usar `weights_only=True` y checksums. [P]
- **B9** (UI) La función de arranque no tiene `try/catch` global; si falla, el splash solo se quita por el temporizador. [P]
- **B10** (Panics) `ort` con `load-dynamic` puede entrar en pánico si `onnxruntime.dll` es de otra versión. [P]

## Orden recomendado

1. Token por sesión + CSP (A3). 2. Restringir `reveal?open` y `upload` (A1). 3. Validar `/api/config` (A2). 4. Corregir `revealBtn` (A4).
5. Job Object, cancelación y timeouts (A5). 6. Temporales y limpieza del video (A6). 7. El resto por orden.

El hardware solo se ha probado en Intel + AMD RX 6600: NVIDIA, Intel Arc y portátiles híbridos siguen sin probarse (afecta sobre todo a M6, M8 y la
selección de GPU).
