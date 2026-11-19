# Ruta: dependencias nativas + instalador de componentes

> Documento de **viabilidad**: no hay código nuevo todavía. Cada fase termina con un criterio claro de «sigue / no sigue»
> para no invertir más de la cuenta si algo resulta inviable. Lo marcado con **(verificar)** es una hipótesis que aún
> no se ha comprobado en este proyecto.

## 0. Resultado de la prueba de viabilidad (Fases 0–4 del Camino A, hecha en aislado)

> Detalle, comandos y mediciones en [`experimental/onnx-faces/README.md`](../experimental/onnx-faces/README.md).

**Veredicto: sí es viable cambiar Python + torch por ONNX.** Se construyó una versión aparte (sin tocar la app) con el
pipeline completo en Rust (`ort` en modo `load-dynamic` + ONNX Runtime/DirectML):

| Fase del plan | Estado | Resultado |
|---|---|---|
| 0. Prueba de concepto ONNX + DirectML | Hecha | GFPGAN corre en la RX 6600 en **0.06 s** (torch CPU: 11 s); `ort` compila con el toolchain GNU |
| 1. Conversión | Hecha | GFPGAN v1.4, **RestoreFormer++**, RetinaFace y ParseNet exportan a ONNX y coinciden con torch (dif. ≤ 0.0007) |
| 2. Pipeline de cara en Rust | Hecha | Detección + alineación + restauración + máscara + pegado; PSNR 33.7 dB (GFPGAN) / 34.6 dB (RF++) frente al puente actual |
| 3. Integración en la app | **Hecha (versión de prueba)** | `face_onnx` en el núcleo, `face::restore` con respaldo ONNX→Python, `face_backend`, doctor y UI; ver el README de la prueba |
| 4. RestoreFormer++ | Hecha (en la prueba) | Misma interfaz que GFPGAN; 0.35 s en GPU |
| 5. Real-ESRGAN en ONNX | No evaluada | Sigue siendo opcional |
| 6. Limpieza | Pendiente | Tras integrar |

Tiempos del pipeline completo (foto 700×464, 2 caras): puente Python **36 s** → Rust+ONNX en GPU **3.5 s** (CPU: 25 s).
Probado en carpeta limpia con `PATH` sin Python: funciona. **No funcionó:** la conversión automática a FP16 degrada
GFPGAN y rompe RestoreFormer++ (los modelos siguen en FP32: ~816 MB). **Probado después:** imágenes grandes (hasta 9000 px), grises, alfa, 16 bits, EXIF, caras en el borde y pequeñas.
**Sigue sin probarse:** GPU NVIDIA/Intel y Windows 10. **Licencias:** GFPGAN incluye componentes no comerciales (ver el README de la prueba).

## 1. Objetivo

Que un usuario **no técnico** abra `Reescaller.exe` y todo funcione, sin instalar Python, torch ni librerías a mano.
Dos caminos que se complementan:

| Camino | Qué resuelve |
|---|---|
| **A. Migración a ONNX nativo** | Elimina Python + torch (lo único realmente pesado y «técnico» que queda) |
| **B. Instalador de componentes en el primer arranque** | La propia app descarga e instala lo que falte, con barra de progreso y sin pedir nada al usuario |

Orden recomendado: **A (Fase 0) → B → resto de A.** B es útil aunque A no salga, pero cuesta mucho menos si A sale
(sin Python, los componentes pasan de ~2 GB y pasos frágiles a unos pocos archivos).

## 2. Punto de partida (qué necesita hoy cada pieza)

| Pieza | Cómo llega hoy al usuario | ¿Requiere instalar algo? |
|---|---|---|
| `Reescaller.exe` (Rust) | Dentro del ZIP | No (usa el CRT universal de Windows 10+) |
| Real-ESRGAN ncnn-Vulkan + modelos ncnn (~45 MB) | Dentro del ZIP | No |
| `vcomp140.dll` | Dentro del ZIP, junto al motor | No |
| FFmpeg (~94 MB) | Dentro del ZIP | No |
| WebView2 (la ventana) | Windows 11 lo trae | **Windows 10 antiguo**: instalador de Microsoft (~2 MB) |
| Driver gráfico con Vulkan | Del sistema | **No se puede empaquetar**; sin él, modo CPU (sin IA) |
| GFPGAN / RestoreFormer++ (614 MB) + Python con torch/basicsr/gfpgan | **No viajan** (solo en la máquina de desarrollo) | **Sí, y es técnico** |

Conclusión: el único bloqueo real para no técnicos es la restauración de rostros.

## 3. Camino A — Rostros con ONNX Runtime (sin Python)

Idea: convertir los modelos de PyTorch a **ONNX** y ejecutarlos desde Rust con el crate `ort`, usando **DirectML**
como backend de GPU (funciona en AMD, NVIDIA e Intel con DirectX 12) y CPU como respaldo.

### Decisiones técnicas previas (verificar)
- **Toolchain GNU vs. binarios de ONNX Runtime**: `ort` descarga por defecto binarios pensados para MSVC. Con el
  toolchain GNU actual lo más seguro es el modo **`load-dynamic`**: no se enlaza nada y se distribuye
  `onnxruntime.dll` (compilación **DirectML**, unos 15–25 MB) + `DirectML.dll` (~18 MB) junto al `.exe`. **(verificar)**
- **Alternativas** si DirectML diera problemas: backend CPU de ONNX Runtime (más lento, pero universal), o `wgpu`
  (`burn`/`candle`), que es más experimental para estos modelos.
- **Licencias**: revisar la de cada peso (GFPGAN, RestoreFormer++, detector de caras) antes de redistribuirlos
  dentro del ZIP o descargarlos desde un servidor propio. **(verificar)**

### Fases

| Fase | Trabajo | Criterio de éxito (go / no-go) |
|---|---|---|
| **0. Prueba de concepto** (½ día) | Cargar un GFPGAN v1.4 en ONNX desde un binario Rust mínimo con `ort` + DirectML; pasar una cara ya alineada de 512×512 y guardar la salida | Corre en la RX 6600, **tarda ≤ ~2× lo que tarda hoy con torch en CPU o menos**, y el resultado es visualmente equivalente. Si no corre o es inutilizable → parar y quedarse con el puente Python + camino B |
| **1. Conversión** | Script único (`scripts/export_onnx.py`, se usa una sola vez en desarrollo) torch → ONNX, opset reciente, ejes dinámicos de batch, versión FP16. Validar contra torch (diferencia numérica/PSNR) | GFPGAN y RestoreFormer++ exportan sin errores y coinciden con torch. Si RestoreFormer++ no exporta limpio, se migra solo GFPGAN |
| **2. Pipeline de cara en Rust** | Detección (RetinaFace u otro detector ONNX) → 5 puntos → alineación afín a 512 px → modelo → máscara con bordes suaves → pegado de vuelta en la imagen original (sin alterar el resto) | La imagen resultante es indistinguible de la del puente Python sobre 10 fotos de prueba (con una, varias y ninguna cara) |
| **3. Integración** | Nuevo backend en `face.rs` (`onnx`), el puente Python queda como respaldo; selección automática; `doctor` y UI de «Modelos y extras» actualizados | Restaurar rostros funciona en una máquina **sin Python** |
| **4. RestoreFormer++** | Igual que la fase 2/3 con su modelo | Calidad equivalente; tiempo aceptable |
| **5. (Opcional) Real-ESRGAN en ONNX** | Reemplazar el `.exe` de ncnn por el mismo motor ONNX/DirectML | **Solo si** iguala la velocidad del ncnn actual en la RX 6600. Si no, se deja ncnn: ya funciona bien |
| **6. Limpieza** | Quitar Python/`venv`/script puente del flujo normal (mantener como «modo desarrollador» si se quiere) | El ZIP + componentes descargados bastan para todas las funciones |

### Riesgos
- Exportar transformers/cuantización vectorial a ONNX a veces falla o genera grafos lentos.
- Diferencias sutiles de calidad por FP16 o por la alineación/pegado (hay que cuidar la máscara).
- Tamaño: DLLs de ONNX Runtime + DirectML suman ~35–45 MB; los modelos en FP16 podrían rondar 150–300 MB en total
  (**estimación, sin medir**).

## 4. Camino B — Instalador de componentes (primer arranque)

La propia app detecta lo que falta y lo instala con un **asistente de configuración**, pensado para no técnicos.

### Experiencia de usuario

```
Primer arranque (o al usar una función que falta)
┌──────────────────────────────────────────────┐
│  Preparemos Reescaller                        │
│  Para usar todo, falta descargar:             │
│   ☑ Motor de mejora de imágenes     45 MB     │
│   ☑ Restaurar rostros               180 MB    │
│   ☐ Conversión y edición de video    94 MB    │
│                                               │
│  [ Instalar ahora ]     [ Más tarde ]         │
│  ▓▓▓▓▓▓▓▓░░░░░  62 %  Descargando…            │
└──────────────────────────────────────────────┘
```
- Un solo botón («Instalar ahora»); sin elegir carpetas ni rutas.
- Barra de progreso, velocidad y tiempo estimado; se puede cancelar y **reanuda** donde quedó.
- Si algo no se puede instalar solo (driver gráfico), se explica en lenguaje llano y se ofrece el enlace del fabricante
  detectado (AMD/NVIDIA/Intel). **Un driver nunca se instala en silencio desde la app.**
- Modo sin conexión: «Elegir archivo…» para componentes descargados antes (también existe hoy como «Elegir…»).

### Diseño técnico

| Tema | Propuesta |
|---|---|
| Manifiesto | `components.json` alojado por HTTPS (p. ej. releases de GitHub): por componente, `id`, versión, URL, tamaño, **SHA-256**, licencia, plataforma y si es obligatorio u opcional |
| Dónde se instala | `%LOCALAPPDATA%\Reescaller\components\<id>\<versión>` (sin permisos de administrador; funciona aunque la app esté en Program Files) |
| Descarga | Módulo `components.rs` en el núcleo: cliente HTTP con TLS (`ureq`/`reqwest` con rustls), reanudación con `Range`, reintentos, verificación SHA-256 antes de activar, descompresión con `zip`, instalación atómica (carpeta temporal → renombrar) |
| Integridad | Hash en el manifiesto; idealmente el manifiesto **firmado** (ed25519/minisign) con clave pública embebida en el `.exe`, para que nadie pueda sustituir descargas |
| API/UI | `GET /api/components` (estado + tamaños), `POST /api/components/install`, progreso por el mismo canal de logs que ya usa la UI; modal del asistente con el estilo actual |
| Detección | Reutiliza `doctor` y la autodetección de rutas, ampliada con la carpeta de componentes |
| Actualizaciones | Comparar versiones del manifiesto; actualizar componentes sin reinstalar la app |
| Privacidad | Solo descargas; sin telemetría. Documentarlo en la pantalla del asistente |

### Qué componentes tendría sentido descargar

| Componente | ¿Incluido en el instalador o descargado? |
|---|---|
| App + motor de mejora + modelos ncnn básicos | **Incluido** (la app funciona en cuanto se abre) |
| FFmpeg (94 MB) | Descargable (aligera el instalador; solo lo necesita Video) |
| Modelos ncnn extra (anime, video) | Descargables |
| Rostros (ONNX + DLLs de ONNX Runtime/DirectML) | Descargables (tras el Camino A) |
| WebView2 | Lo resuelve el instalador de Windows (Inno Setup) con el bootstrapper de Microsoft, no la app |
| Python + torch | **No se ofrece** a usuarios finales (≈2 GB, frágil). Solo para desarrolladores |

Dos variantes de instalador: **Ligera** (~10–50 MB, descarga el resto al primer uso) y **Completa/offline** (todo dentro).

### Fases

| Fase | Trabajo | Criterio de éxito |
|---|---|---|
| **B0. Manifiesto + descargador** | `components.rs` con descarga, hash, reanudación y activación atómica; pruebas con un archivo pequeño | Descarga interrumpida y reanudada correctamente; hash incorrecto rechazado |
| **B1. Asistente en la UI** | Modal del primer arranque + estado en «Modelos y extras» | Un usuario sin conocimientos instala y usa la app sin ver rutas |
| **B2. Mover FFmpeg y modelos extra a «descargables»** | Ajustar `build_dist.ps1` y el instalador (variante ligera) | ZIP/instalador ligero funcional; la app pide lo que falte al usarlo |
| **B3. Rostros** | Publicar los modelos ONNX como componente (depende del Camino A) | Restaurar rostros funciona en un PC limpio tras un clic |
| **B4. Firma y actualizaciones** | Manifiesto firmado, comprobación de versiones | Un manifiesto manipulado es rechazado |

## 5. Orden de trabajo sugerido y estimación

| # | Paso | Esfuerzo orientativo |
|---|---|---|
| 1 | **A · Fase 0** (prueba de concepto ONNX + DirectML) | ½ día |
| 2 | Decisión go / no-go de A | — |
| 3 | **B0 + B1** (descargador + asistente) | 1–2 sesiones |
| 4 | **A · Fases 1–3** (GFPGAN nativo) | 1–2 sesiones |
| 5 | **B2 + B3** (componentes descargables, rostros) | 1 sesión |
| 6 | **A · Fase 4** (RestoreFormer++) y **B4** (firma) | 1 sesión cada uno |
| 7 | A · Fase 5 (Real-ESRGAN ONNX) — solo si mejora | opcional |

Las estimaciones son orientativas, no mediciones.

## 6. Preguntas abiertas (necesito tu decisión)

1. **¿Dónde alojar los archivos descargables?** (releases de GitHub es lo más simple y gratuito.)
2. **Tamaño máximo aceptable** del instalador ligero y de los componentes.
3. **Licencias**: confirmar que se pueden redistribuir los pesos de GFPGAN, RestoreFormer++, el detector de caras y
   el FFmpeg elegido.
4. **Versiones de Windows objetivo** (¿Windows 10 incluido? Afecta a WebView2 y DirectML).
5. ¿La variante **offline completa** es necesaria o basta con la ligera?
