# Avisos de terceros y licencias

Reescaller (código propio, licencia MIT: ver [`LICENSE`](LICENSE)) acompaña, enlaza o ejecuta software y modelos de otros autores. Este documento los lista, indica su licencia y lo que exige cada una. Los textos originales están en la carpeta [`licenses/`](licenses/) y se instalan junto a la aplicación.

> Es una revisión técnica hecha con las fuentes de cada proyecto, **no asesoría legal**. Quien redistribuya o use Reescaller con fines comerciales debe confirmar las condiciones con un profesional; ver «Uso comercial» más abajo.

## Qué se distribuye

| Componente | Para qué sirve | Licencia | Texto |
|---|---|---|---|
| **FFmpeg 8.0** (build *essentials* de gyan.dev) — `models\ffmpeg.exe` | Todo el video: extraer, codificar, convertir | **GPL v3** (build con `--enable-gpl --enable-version3`, incluye x264 y x265) | `licenses/FFmpeg-COPYING.GPLv3.txt`, `FFmpeg-LICENSE.md` |
| **Real-ESRGAN ncnn Vulkan** — `models\realesrgan-ncnn-vulkan.exe` | Motor de mejora de imágenes por GPU | MIT © 2021 Xintao Wang | `licenses/Real-ESRGAN-ncnn-vulkan-MIT.txt` |
| **Modelos Real-ESRGAN** (`realesrgan-x4plus`, `realesrgan-x4plus-anime`, `realesr-animevideov3`) — `models\models\*.bin\|param` | Pesos de la mejora | BSD-3-Clause © 2021 Xintao Wang (proyecto Real-ESRGAN) | `licenses/Real-ESRGAN-BSD-3-Clause.txt` |
| **GFPGAN v1.4** (convertido a ONNX) — `models\onnx\gfpgan_v14.onnx` | Restauración de rostros | Apache-2.0 © Tencent, **con componentes de terceros de uso no comercial** (ver abajo) | `licenses/GFPGAN-LICENSE.txt` |
| **RestoreFormer++** (convertido a ONNX) — `models\onnx\restoreformer_pp.onnx` | Restauración de rostros | Apache-2.0 | `licenses/RestoreFormerPlusPlus-Apache-2.0.txt` |
| **RetinaFace ResNet50** (de *Pytorch_Retinaface*, vía facexlib) — `retinaface_r50.onnx` | Detección de rostros | MIT | `licenses/Pytorch_Retinaface-MIT.txt` |
| **ParseNet** (de *face-parsing.PyTorch*, vía facexlib) — `parsenet.onnx` | Máscara del rostro | MIT | `licenses/face-parsing.PyTorch-MIT.txt`, `facexlib-MIT.txt` |
| **ONNX Runtime 1.24** — `onnxruntime.dll` | Ejecuta los modelos de rostros | MIT © Microsoft | `licenses/ONNXRuntime-LICENSE.txt`, `ONNXRuntime-ThirdPartyNotices.txt`, `ONNXRuntime-Privacy.md` |
| **DirectML 1.15.4** — `DirectML.dll` | Aceleración por GPU en Windows | Microsoft Software License Terms (se puede redistribuir dentro de aplicaciones que corren en Windows) | `licenses/DirectML-Microsoft-License.txt`, `DirectML-ThirdPartyNotices.txt` |
| **Visual C++ OpenMP** — `vcomp140.dll` | Lo necesita el motor de mejora | Microsoft (runtime redistribuible de Visual C++) | — |
| **Microsoft Edge WebView2** — `WebView2Loader.dll` y el instalador del Runtime | Ventana de la aplicación | Microsoft (redistribuible, instalador oficial firmado) | — |
| **Crates de Rust** (227 dependencias) | Núcleo de la aplicación | MIT, Apache-2.0, BSD, ISC, Zlib, Unicode-3.0, BSL-1.0, y una MPL-2.0 (`option-ext`, sin modificar) | `licenses/RUST_CRATES.txt` |

Los crates `webview2-com*` (MIT, © Microsoft y colaboradores) y algunos otros no traen archivo de licencia propio en su paquete; su licencia SPDX consta en `licenses/RUST_CRATES.txt`.

## FFmpeg (GPL v3): código fuente y condiciones

`ffmpeg.exe` se ejecuta como **programa independiente** (Reescaller lo llama por línea de comandos), por lo que el resto de la aplicación conserva su licencia MIT. Al redistribuirlo se cumple con:

- **Texto de la licencia:** `licenses/FFmpeg-COPYING.GPLv3.txt`.
- **Código fuente correspondiente:** FFmpeg 8.0 está en <https://ffmpeg.org/releases/ffmpeg-8.0.tar.xz> y se adjunta una copia a cada *release* de Reescaller (SHA-256 `b2751fccb6cc4c77708113cd78b561059b6fa904b24162fa0be2d60273d27b8e`). La build empaquetada es la de gyan.dev (<https://www.gyan.dev/ffmpeg/builds/>), compilada con MinGW-w64/GCC 15.2 y estas bibliotecas, también bajo GPL/LGPL y de código abierto: x264, x265, libvpx, libaom, libopus, libvorbis, libmp3lame, libwebp, libass, libfreetype, libxvid, libzimg, libvmaf y otras listadas por `ffmpeg -version`. Su código fuente está disponible en los sitios de cada proyecto.
- **Oferta por escrito:** quien reciba Reescaller puede pedir el código fuente completo correspondiente a ese `ffmpeg.exe` abriendo un *issue* en <https://github.com/BryanZzzz1/Reescaller-App-/issues>; se entregará durante al menos tres años desde la publicación de la versión.
- **Puedes sustituirlo:** `ffmpeg.exe` es un archivo suelto en `models\`; se puede reemplazar por otra compilación de FFmpeg.
- **Patentes de códecs:** H.264/H.265 y otros formatos pueden estar sujetos a patentes en algunos países; las licencias de software libre no las cubren.

## Modelos de rostros: uso comercial

- **GFPGAN**: su licencia (`licenses/GFPGAN-LICENSE.txt`) es Apache-2.0 **salvo los componentes de terceros que lista**, entre ellos código derivado de **StyleGAN2** (licencia NVIDIA, solo uso no comercial) y de **DFDNet** (CC BY-NC-SA 4.0, no comercial). La arquitectura «clean» de GFPGAN v1.4 se basa en StyleGAN2, y los pesos se entrenaron con **FFHQ**, cuyo conjunto de datos tiene licencia de uso no comercial. Los autores no aclaran si esas restricciones alcanzan a los pesos publicados.
- **RestoreFormer++** declara Apache-2.0, pero también se entrenó con FFHQ.
- **Conclusión práctica:** para uso **personal, de investigación o sin ánimo de lucro** no hay impedimento conocido. Si vas a **vender** Reescaller o a ofrecer un servicio comercial con la restauración de rostros, consulta antes con los autores o con un abogado, o desactiva ese componente (el instalador permite omitirlo: opción «Restauración de rostros»). Los modelos de Real-ESRGAN, el resto del programa y las dependencias de Rust no tienen esta restricción.

## Privacidad de componentes de Microsoft

ONNX Runtime y DirectML pueden recopilar datos de uso según sus licencias (`ONNXRuntime-Privacy.md`). Reescaller no envía imágenes ni archivos a ningún servidor; consulta esos documentos para el detalle de la telemetría de Microsoft.

## Regenerar la lista de dependencias de Rust

```
python -I scripts\gen_third_party.py
```
