# Reescaller en Rust

Port a Rust de la *Herramienta de Mejora Visual* (CustomTkinter → Rust), con una
**interfaz de depuración** para probar todas las funcionalidades.

```
rust/
├── Cargo.toml                        # workspace
├── build.bat / build.sh              # helpers de compilación (Windows + Git Bash)
└── crates/
    ├── reescaller-core/              # dominio puro (sin UI, sin dependencias de GUI)
    │   ├── src/config.rs             # AppConfig, Weight, ResizeMode, autodetección
    │   ├── src/process.rs            # ejecución de procesos + detección de FPS
    │   ├── src/image_ops.rs          # resize, blend, sharpen, encode (peso)
    │   ├── src/upscale.rs            # pipeline de reescalado controlado
    │   ├── src/batch.rs              # procesamiento por lotes
    │   ├── src/video.rs              # extract → enhance → mux
    │   ├── src/face.rs               # puente a GFPGAN / RestoreFormer++
    │   ├── src/remaster.rs           # modo fácil: análisis de la imagen + presets
    │   ├── src/hardware.rs           # autodetección CPU/RAM/GPU (NVIDIA/AMD/Intel) y Vulkan
    │   └── src/diagnostics.rs        # "doctor" del entorno
    ├── reescaller-cli/               # CLI (binario `reescaller`)
    ├── reescaller-app/               # aplicación de escritorio + servidor de depuración
    │   ├── src/bin/reescaller-app.rs # Reescaller.exe: ventana propia, sin consola
    │   ├── src/main.rs               # reescaller-debug.exe: con consola y navegador
    │   └── assets/index.html         # interfaz (embebida en el ejecutable)
    └── reescaller-selftest/          # verificador automático (29 comprobaciones)
```

---

## 1. Novedades frente a la app Python

| Tema | Antes (Python) | Ahora (Rust) |
|---|---|---|
| **Cuándo reescalar** | Siempre a la máxima del motor | `resize_mode`: `always`, `only_if_needed`, `improve_only` |
| **Cuánto reescalar** | Solo x2 / x4 del motor | `target_scale` continuo (0.25–8) → el motor escala a x4 y el resultado se ajusta al tamaño elegido |
| **Mejora por porcentaje** | No existía | `improve_percent` (0–100): mezcla entre la original reescalada y la mejorada |
| **Peso de la imagen** | Solo formato PNG/JPG | `Weight::Quality { quality }` o `Weight::TargetSize { target_kb }` (búsqueda binaria de la calidad que más se acerca al tamaño objetivo) |
| **Nitidez extra** | No existía | `sharpen` (0–100), unsharp mask 3×3 |
| **Filtro de reescalado** | No existía | Lanczos3 / Catmull-Rom / Bilinear / Nearest |
| **Lote** | Hilos con `os.cpu_count()` | Workers configurables (1–16) con progreso real |
| **Video** | FPS fijo a 23.98 | Hereda el FPS real del origen (`detect_fps`) |
| **Errores** | `except: pass` | Errores tipados (`thiserror`) propagados a la UI |
| **Configuración** | JSON en Python | JSON en Rust (`%APPDATA%\Reescaller\config.json`), mismo formato de datos clave |
| **Diagnóstico** | Manual | `reescaller doctor` + pestaña Diagnóstico |

### Cómo se comporta el reescalado

```
original 400x300
   │
   ├─ motor Real-ESRGAN  →  1600x1200   (x4, su máximo)
   │
   ├─ reescalado al objetivo elegido  →  600x450   (target_scale = 1.5)
   │
   ├─ mezcla con la original           →  improve_percent = 100 → 80
   │
   ├─ nitidez                          →  sharpen = 20
   │
   └─ codificación con peso            →  JPEG calidad 88  (o "ajustar a 250 KB")
```

---

## 2. Requisitos

| | |
|---|---|
| **Rust** | 1.75+ (probado con 1.100). Toolchain **GNU** en Windows: `rustup default stable-x86_64-pc-windows-gnu` |
| **MinGW-w64** | Necesario para `dlltool.exe` (se usa `cc`/binutils). En esta máquina: WinLibs (`winget install BrechtSanders.WinLibs.POSIX.UCRT`) |
| **Modelos** | `models/ffmpeg.exe`, `models/realesrgan-ncnn-vulkan.exe`, `models/GFPGANv1.4.pth`, `models/RestoreFormer++.ckpt` |
| **Python (opcional)** | Solo para el puente de rostros (`venv/Scripts/python.exe` + `scripts/restore_face.py`) |

---

## 3. Compilar

```bat
:: Windows (recomendado: prepara el PATH con MinGW y llama a cargo)
cd F:\rescaller\Reescaller-App-\rust
build.bat build --release
```

```bash
# Git Bash
cd /f/rescaller/Reescaller-App-/rust
./build.sh build --release
```

> **Nota sobre Git Bash**: MSYS corrompe las rutas con barras mixtas al pasarlas a
> procesos nativos, y `rustc` no encontraba `dlltool.exe`. Por eso `build.sh`/
> `build.bat` reconstructen el `PATH` en formato Windows nativo antes de invocar
> a `cargo`.

Binarios resultantes:

```
rust/target/release/reescaller.exe         # CLI
rust/target/release/reescaller-debug.exe   # interfaz de depuración
```

---

## 4. Verificador automático (cómo probarlo en tu equipo)

Desde la raíz del proyecto:

```bat
probar.bat
```

Compila (si hace falta) y ejecuta **18 comprobaciones** sobre tu entorno, con
informe PASS/FAIL:

```
 PASA entorno · Real-ESRGAN                                .\models
ealesrgan-ncnn-vulkan.exe
 PASA entorno · FFmpeg                                     .\modelsfmpeg.exe
 PASA entorno · Modelo GFPGAN                              .\models\GFPGANv1.4.pth
 PASA entorno · Modelo RestoreFormer++                     .\models\RestoreFormer++.ckpt
 PASA entorno · Puente Python (torch+gfpgan)               OK: torch + gfpgan + cv2 + numpy importan
 PASA test · imagen de muestra generada                    600x400
 PASA reescalado · escala elegida x1.5 (no la máxima del motor) 900x600 (esperado 900x600)
 PASA reescalado · modo 'solo mejora' conserva el tamaño   600x400
 PASA peso · objetivo 30 KB (búsqueda de calidad)          30.0 KB en 1200x800
 PASA reescalado · modo rápido sin motor de IA             1200x800
 PASA lote · carpeta procesada                             2 correctas, 0 fallidas
 PASA video · clip de prueba generado (FFmpeg)             320x240 @25fps con audio
 PASA video · escala 1.5 aplicada a los frames             480x360 (esperado 480x360)
 PASA video · FPS heredado del original (25)               25
 PASA video · audio original conservado                    50 frames
 PASA interfaz · servidor de depuración                    respondiendo en http://127.0.0.1:7878
──────────────────────────────────────────────────────────────
 Resultado: 18/18 comprobaciones superadas
 ✔ Todo correcto. El port a Rust funciona en este equipo.
```

El verificador **no necesita nada externo**: genera su propia imagen de prueba,
crea un video con FFmpeg y usa `models/` para los motores. No toca tus archivos
personales (limpia `selftest_tmp/` al terminar; usa `--keep` para conservarlo).

Equivalente sin el `.bat`:

```bash
rust/target/release/reescaller-selftest.exe --project . [--keep]
```

Código de salida: `0` si todo pasa, `1` si hay algún fallo (útil para CI).

---

## 5. CLI

```bash
# Diagnóstico del entorno
reescaller doctor

# Ver la configuración actual
reescaller config

# Reescalar una imagen al 150 % (el motor trabaja a x4 y se ajusta al final)
reescaller upscale --input foto.jpg --target 1.5 --improve 100 --format jpg

# Reescalar ajustando el peso del archivo a 250 KB (busca la calidad óptima)
reescaller upscale --input foto.jpg --target 2 --target-size-kb 250 --format jpg

# Solo mejorar, conservando el tamaño original
reescaller upscale --input foto.jpg --mode improve-only

# Sin motor de IA (reeescalado rápido, útil para pruebas)
reescaller upscale --input foto.jpg --target 2 --no-engine

# Lote
reescaller batch --input entrada --output salida --target 2 --workers 2

# Video (hereda el FPS original)
reescaller video --input clip.mp4 --output clip_mejorado.mp4 --target 1.5

# Rostros (puente a Python)
reescaller face --input cara.jpg --model gfpgan
```

Opciones destacadas de `upscale`:

| Flag | Significado |
|---|---|
| `--engine 2\|4` | Escala del motor (solo el modelo `realesr-animevideov3` admite x2) |
| `--target 1.75` | Escala final respecto al original (0.25–8) |
| `--improve 80` | % de la imagen mejorada que se conserva |
| `--weight-quality 88` | Calidad de codificación (1–100) |
| `--target-size-kb 250` | Peso objetivo del archivo |
| `--mode always\|only-if-needed\|improve-only` | Cuándo reescalar |
| `--sharpen 20` | Nitidez extra (0–100) |
| `--format png\|jpg` | Formato de salida |
| `--no-engine` | No usar el motor de IA |

---

## 6. Interfaz de depuración

```bash
cd F:/rescaller/Reescaller-App-
rust/target/release/reescaller-debug.exe --port 7878 --project .
```

Y abrir <http://127.0.0.1:7878>.

**Botón de ejecución:** la pestaña *Imagen* tiene una **barra de acción fija
(sticky)** arriba con el botón `🚀 Procesar imagen` (y su variante `Sin motor de
IA`), además del botón en la tarjeta *Comparativa* y en el plan. Mientras procesa
muestra un spinner y deshabilita los botones; al terminar muestra un aviso
emergente con el tamaño y el peso del resultado. Atajo: **Ctrl + Enter**.

**Pestañas:**

1. **Imagen** — entrada (ruta, explorador integrado o subida), plan completo
   (escala del motor, escala final, cuándo reescalar, % mejora, peso, nitidez,
   modelo, filtro, formato) y comparativa antes/después con el factor real y el
   peso del archivo resultantes.
2. **Lote** — carpetas de entrada/salida y workers.
3. **Video** — modelo y escala final de los frames.
4. **Rostros** — GFPGAN / RestoreFormer++ vía puente Python.
5. **Diagnóstico** — checklist del entorno + configuración en JSON.

### API HTTP

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | UI |
| GET | `/api/config` | Configuración actual |
| POST | `/api/config` | Guarda la configuración |
| GET | `/api/doctor` | Diagnóstico del entorno |
| GET | `/api/logs?since=N` | Líneas del registro |
| GET | `/api/browse?path=` | Explorador de archivos |
| GET | `/api/preview?path=&max=` | Miniatura en base64 |
| POST | `/api/upload` | Subida de imagen (base64) |
| POST | `/api/upscale` | `{ input, output?, plan }` |
| POST | `/api/batch` | `{ input, output, plan, workers }` |
| POST | `/api/video` | `{ input, output, model, engine_scale, target_scale }` |
| POST | `/api/face` | `{ input, output?, model }` |

Ejemplo:

```bash
curl -X POST http://127.0.0.1:7878/api/upscale \
  -H "Content-Type: application/json" \
  -d '{
    "input": "F:/.../foto.jpg",
    "plan": {
      "engine_scale": 4,
      "model": "realesrgan-x4plus",
      "target_scale": 1.75,
      "improve_percent": 90,
      "weight": { "mode": "target_size", "target_kb": 250 },
      "output_format": "jpg",
      "resize_mode": "always",
      "resize_filter": "lanczos3",
      "sharpen": 10,
      "use_engine": true
    }
  }'
```

---

## 7. Puente de rostros (Python)

Los modelos de rostros (GFPGAN, RestoreFormer++) siguen siendo PyTorch. La app
en Rust los invoca con `scripts/restore_face.py`:

```bash
venv/Scripts/python.exe scripts/restore_face.py \
  --model gfpgan --weights models/GFPGANv1.4.pth \
  --input cara.jpg --output cara_mejorada.png
```

El script aplica el parche de compatibilidad de `torchvision`
(`torchvision.transforms.functional_tensor`, eliminado en torchvision ≥ 0.17),
necesario para importar `basicsr`/`gfpgan`/`facexlib`.

**Hoja de ruta**: exportar GFPGAN y RestoreFormer++ a ONNX y ejecutarlos con
`ort` (crate de Rust), eliminando por completo la dependencia de Python.

---

## 8. Estructura de la configuración

`%APPDATA%\Reescaller\config.json` (Windows) o `~/.config/Reescaller/config.json`:

```json
{
  "realesrgan_exe": "...\\models\\realesrgan-ncnn-vulkan.exe",
  "ffmpeg_exe": "...\\models\\ffmpeg.exe",
  "gfpgan_model": "...\\models\\GFPGANv1.4.pth",
  "restoreformer_model": "...\\models\\RestoreFormer++.ckpt",
  "python_exe": "...\\venv\\Scripts\\python.exe",
  "engine_scale": 4,
  "model": "realesrgan-x4plus",
  "target_scale": 2.0,
  "improve_percent": 100,
  "weight": { "mode": "quality", "quality": 92 },
  "output_format": "png",
  "resize_mode": "always",
  "resize_filter": "lanczos3",
  "sharpen": 0,
  "delete_temp": true,
  "theme": "Dark",
  "max_workers": 2
}
```

---

## 9. Estado del port

| Módulo | Python | Rust | Estado |
|---|---|---|---|
| Configuración + autodetección | `All.py` | `core/config.rs` | ✅ |
| Ejecutar Real-ESRGAN / FFmpeg | `proceso_ejecutar` | `core/process.rs` | ✅ |
| Mejora de imagen | `ejecutar_realesrgan_unico` | `core/upscale.rs` | ✅ + escala/peso controlables |
| Lote de carpeta | `ejecutar_realesrgan_carpeta` | `core/batch.rs` | ✅ |
| Pipeline de video | `ejecutar_proceso_video` | `core/video.rs` | ✅ + FPS heredado |
| Conversión / recorte | `convertir_video`, `recortar_video` | `core/video.rs` | ✅ |
| Rostros (GFPGAN/RF++) | `ejecutar_gfpgan` | `core/face.rs` + puente Python | ✅ (puente) |
| Visualizador de video | `reproducir_video` | — | ⏳ pendiente (egui) |
| GUI principal (CustomTkinter) | `All.py` | — | ⏳ pendiente (egui/eframe) |
| Diagnóstico | `verificar_dependencias_inicio` | `core/diagnostics.rs` | ✅ |
