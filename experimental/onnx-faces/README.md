# onnx-faces · rostros sin Python ni torch (prueba de viabilidad)

Versión **aparte** de la app: no toca `Reescaller.exe`, su `venv` ni su configuración. Responde a la pregunta
«¿se pueden sustituir Python + torch por ONNX?». **Resultado: sí, en este equipo funciona de punta a punta.**

## Qué hay

```
experimental/onnx-faces/
├── export_onnx.py      torch -> ONNX (GFPGAN, RestoreFormer++, RetinaFace, ParseNet). Se usa UNA vez, en desarrollo
├── validate.py         compara torch vs ONNX Runtime (CPU y DirectML) red por red
├── make_edge_cases.py  genera las imágenes de casos límite (gris, alfa, 16 bits, borde, EXIF, 4000/6000/9000 px)
├── rust/               pipeline completo en Rust (crate `onnx-faces`, proyecto independiente)
├── build.bat           compila `rust/` con el mismo toolchain GNU de la app
├── bin/                onnxruntime.dll + DirectML.dll (ONNX Runtime 1.24.4, build DirectML)  [ignorado por git]
├── models/             *.onnx exportados (~816 MB en FP32)                                    [ignorado por git]
├── pydeps/             onnx / onnxruntime-directml solo para exportar y validar               [ignorado por git]
├── samples/ out/       fotos de prueba públicas (GFPGAN) y resultados                         [ignorado por git]
```

## Cómo reproducirlo

```powershell
# 1) exportar los modelos (usa el venv actual con torch; las libs extra van a experimental\onnx-faces\pydeps)
venv\Scripts\python.exe experimental\onnx-faces\export_onnx.py            # gfpgan retinaface parsenet [restoreformer]
venv\Scripts\python.exe experimental\onnx-faces\validate.py               # torch vs ONNX

# 2) compilar y ejecutar el pipeline Rust (sin Python)
experimental\onnx-faces\build.bat build --release
cd experimental\onnx-faces\rust
.\target\release\onnx-faces.exe --input ..\samples\Blake_Lively.jpg --output ..\out\resultado.png
#   opciones: --model gfpgan|restoreformer   --cpu   --upscale 2   --models <dir>   --dll <onnxruntime.dll>
```

## Pipeline en Rust (réplica de facexlib + GFPGANer)

1. **RetinaFace** (ONNX, tamaño de entrada dinámico) → caras + 5 puntos; anclas, decodificación y NMS en Rust.
2. **Alineación** afín a la plantilla de 512 px (similitud por mínimos cuadrados) y `warpAffine` propio.
3. **GFPGAN v1.4 / RestoreFormer++** (ONNX) sobre la cara de 512×512.
4. **ParseNet** (ONNX) → máscara de la cara, desenfoque gaussiano 101×101 σ=11 (igual que facexlib).
5. **Pegado** sobre la imagen ampliada (Lanczos) con la afín inversa.

Dependencias de ejecución: **`onnx-faces.exe` + `onnxruntime.dll` + `DirectML.dll`** (≈ 38 MB) + los `.onnx`. Nada más.

## Resultados (Intel i5-9600KF + AMD RX 6600)

### Las redes coinciden con torch
| Red | Diferencia máxima vs torch | Tiempo (512×512) |
|---|---|---|
| GFPGAN v1.4 | 0.0000 (CPU y DirectML) | torch CPU 11.0 s · ONNX CPU 4.7 s · **ONNX DirectML 0.06 s** |
| RestoreFormer++ | 0.0007 (CPU) / 0.0001 (DirectML) | torch CPU 6.2 s · ONNX CPU 9.3 s · **ONNX DirectML 0.35 s** |
| RetinaFace R50 | ≤ 0.0001, también con 640×640 y 720×480 | — |
| ParseNet | 100 % de etiquetas iguales | — |

### Pipeline completo, foto de 700×464 con 2 caras (salida 1400×928)
| Camino | Tiempo total |
|---|---|
| Puente actual (Python + torch, CPU) | **36 s** (GFPGAN) · 32 s (RestoreFormer++) |
| Rust + ONNX, **GPU DirectML** | **3.5 s** (GFPGAN, 2.2 s son de carga de modelos) · 4.8 s (RestoreFormer++) |
| Rust + ONNX, solo CPU | 25.5 s |

Calidad frente al puente (misma foto): **PSNR 33.7 dB (GFPGAN) y 34.6 dB (RestoreFormer++)**; a simple vista
indistinguibles (`out/comparacion.png`, `out/comparacion_rf.png`). Las diferencias residuales vienen del ruido
aleatorio que usa torch en GFPGAN (aquí fijo) y de usar interpolación bilineal en vez de bicúbica al pegar la máscara.
Una foto con 3 caras (783×1146 → 1566×2292) tardó 4.3 s; una imagen sin caras devuelve el mismo error que hoy.

### Portabilidad
Ejecutado desde una carpeta nueva con solo `onnx-faces.exe` + las 3 DLL y `PATH=C:\Windows\System32` (sin Python):
funciona en GPU (4.4 s) y en CPU (25.5 s).

### Lo que NO funcionó
- **FP16 automático** (`onnxconverter-common`) reduce los modelos a la mitad (816 → 396 MB) pero **degrada GFPGAN**
  (PSNR 9.5 dB frente a FP32) y **RestoreFormer++ ni carga**. No sirve tal cual; haría falta convertir capa por capa
  (listas de operadores a mantener en FP32) o cuantizar con calibración. Se descartó y se borraron esos archivos.

## Veredicto

**Viable.** Con `ort` en modo `load-dynamic` (funciona con el toolchain GNU de la app) y ONNX Runtime + DirectML, la
restauración de rostros se ejecuta en Rust, sin Python ni torch, con calidad equivalente y ~10× más rápido que el
puente actual en GPU.

## Integración en la app y pruebas de robustez

El pipeline ya está **integrado en el núcleo** (`reescaller-core::face_onnx` + `face::restore`) y en la app/CLI:
`face_backend = auto` (por defecto) usa ONNX si encuentra `models\onnx` y, si no o si falla, el puente de Python;
`onnx` fuerza el nativo; `python` fuerza el antiguo. `reescaller-cli face --backend onnx|python|auto`.
Versión de prueba empaquetada con `scripts\build_dist.ps1 -WithOnnx -NoZip -DistRoot dist\prueba-onnx-2`.

Mejoras sobre la prueba de concepto: detección sobre una copia reducida (≤ 1280 px), pegado solo en el rectángulo de
cada cara (sin tocar toda la imagen), recortes muestreados directamente del original, límite de 120 MP de salida,
orientación EXIF, sesiones ONNX reutilizadas entre imágenes y **respaldo GPU → CPU** (al cargar y si falla en ejecución).
`REESCALLER_DML_DEVICE=N` elige el adaptador DirectML (portátiles con dos GPU).

### Resultados (Intel i5-9600KF + AMD RX 6600, GFPGAN salvo indicación)
| Caso | Resultado |
|---|---|
| Foto 700×464, 2 caras | OK · 6.1 s en frío (con carga de modelos) · 1400×928 |
| Escala de grises (JPEG de 1 canal) | OK · 3.2 s · sale en gris |
| PNG con canal alfa | OK · 4.9 s |
| PNG de 16 bits | OK · 6.0 s |
| Cara cortada por el borde | OK · detecta la cara completa y deja intacta la cortada; sin artefactos ni fallos |
| Caras pequeñas (imagen de 300 y 160 px) | OK · 2 caras detectadas en ambos |
| JPEG vertical con EXIF «girar 90°» | OK · sale con la orientación correcta |
| Foto con 3 caras (783×1146) | OK · 5.3 s |
| Foto de 4000 px (salida 8000×5302) | OK · 17.2 s · pico de memoria 827 MB |
| Foto de 6000 px (salida 12000×7954, 95 MP) | OK · 41.2 s · pico 1.27 GB |
| Foto de 9000 px | OK · ampliación reducida a ×1 por el límite de 120 MP · 8.3 s · pico 982 MB |
| DirectML forzado a fallar (adaptador inexistente) | OK · cae a CPU y termina · 31.7 s · pico 1.87 GB |
| RestoreFormer++ (cara en el borde) | OK · 8.1 s |
| Desde la app real (`/api/face`), GFPGAN y RestoreFormer++ | OK · `backend=onnx · DirectML (GPU)` · 7 s |

### Sigue sin probarse
- **GPU NVIDIA / Intel con DirectML** y **Windows 10 / sin DirectX 12**: no hay ese hardware aquí. El respaldo a CPU está
  probado forzando el fallo; con otra GPU habrá que confirmar velocidad y selección de adaptador (`REESCALLER_DML_DEVICE`).
- **Fallo real por falta de memoria de la GPU en ejecución** (solo se probó el fallo al cargar).
- Calidad frente a torch en caras muy ladeadas, con gafas, oclusiones, etc. (solo se compararon dos fotos: PSNR ≈ 34 dB).

### Licencias (lo que dicen los archivos; no es asesoría legal)
| Componente | Licencia encontrada | Observación |
|---|---|---|
| GFPGAN (código y pesos) | Apache 2.0 **«excepto los componentes de terceros»** | Su `LICENSE` lista código de **StyleGAN2 con la licencia de NVIDIA (uso no comercial)** y **DFDNet (CC BY-NC-SA 4.0)** → **riesgo para un uso comercial** |
| RestoreFormer++ | Apache 2.0 (archivo `LICENSE` del repositorio) | Entrenado con FFHQ (dataset con licencias propias, en parte no comerciales); el README no aclara el uso de los pesos |
| facexlib (RetinaFace, ParseNet) | MIT (código); detector y parsing derivan de proyectos MIT | Su README remite a las licencias originales de cada modelo |
| ONNX Runtime | MIT (metadatos de la rueda) | — |
| DirectML.dll | No se pudo verificar: la distribución no incluye su texto de licencia | Confirmar las condiciones de redistribución de Microsoft |

Conclusión práctica: para **uso personal o gratuito** con atribución no hay impedimentos evidentes; antes de **vender o
distribuir comercialmente** hay que revisar GFPGAN (componentes no comerciales), los datos de entrenamiento y DirectML.

### Hallazgo aparte (no corregido)
El camino de remasterizado de imágenes (Real-ESRGAN) tampoco aplica la orientación EXIF: una foto de móvil guardada
«de lado» con etiqueta de giro saldría girada respecto a lo que muestra el visor. Solo se corrigió en rostros.

## Pendiente

- Probar en NVIDIA / Intel y Windows 10; fallo de memoria real en GPU; más fotos de calidad frente a torch.
- Instalador de componentes que descargue estos archivos (`docs/RUTA_MIGRACION_NATIVA.md`, camino B).
- FP16 correcto (la conversión automática degrada GFPGAN) para reducir los ~816 MB de modelos.
- Revisar licencias antes de cualquier uso comercial.
