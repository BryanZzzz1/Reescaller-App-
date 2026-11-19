# Roadmap de Reescaller

Estado actual: **v0.3.0** (app de escritorio en Rust, instalador propio, restauración de retratos con ONNX). Este documento ordena lo que falta antes de una versión estable y lo que se quiere hacer después. El detalle técnico de lo ya hecho está en [`PROJECT.md`](PROJECT.md).

Leyenda: 🔴 bloquea una versión pública · 🟡 importante · 🟢 deseable

## v0.3.x — Estabilizar (antes de publicar)

**Pruebas que solo se pueden hacer en equipos reales**
- 🔴 Probar el instalador con clics en una máquina limpia: opciones, selector de carpeta, presentación, «Abrir Reescaller», desinstalar.
- 🔴 Windows 10 (1809) sin WebView2: comprobar que el instalador lo añade y la app arranca.
- 🔴 GPU **NVIDIA** (GTX 600+), **Intel Arc / iGPU** y portátiles con gráfica **híbrida**: detección, mejora por IA y restauración de rostros con DirectML. Hoy solo está probado Intel + AMD RX 6600.
- 🟡 Recorrer con clics los botones de ventana (minimizar, cerrar), el arrastre y la confirmación al cerrar con un trabajo en curso.

**Distribución**
- 🔴 Añadir el archivo `LICENSE` (el workspace declara MIT) y revisar las licencias de terceros: Real-ESRGAN, FFmpeg (según la compilación incluida), GFPGAN, RestoreFormer++, ONNX Runtime y DirectML. Incluir un `THIRD_PARTY_NOTICES`.
- 🟡 **Firma de código** del instalador, el desinstalador y `Reescaller.exe` para evitar el aviso de SmartScreen.
- 🟡 Publicar la *release* (instalador + ZIP portable + `models/` como descarga aparte) y mover los activos pesados fuera del repositorio.
- 🟢 Incluir `vc_redist.x64.exe` como respaldo si el equipo no tiene el runtime de Visual C++.

**Errores conocidos**
- 🟡 El PNG de salida pierde el canal alfa (transparencia).
- 🟡 En lotes, `a.png` y `a.jpg` generan el mismo nombre de salida y pueden pisarse.
- 🟢 El selector de carpeta del instalador usa PowerShell; sustituirlo por el diálogo nativo de Windows.

## v0.4 — Calidad y comodidad

- 🟡 Cola de trabajos con pausa, cancelación y reanudación, y progreso por archivo en lotes.
- 🟡 Edición de video básica: **recortar clip** (hoy desactivado) y elegir la parte a mejorar.
- 🟡 Más control en retratos: seleccionar el modelo de rostro por foto y vista previa rápida antes de procesar todo el lote.
- 🟢 Actualizador integrado: avisar de nuevas versiones y descargarlas.
- 🟢 Integración con el menú contextual de Windows («Mejorar con Reescaller»).
- 🟢 Soporte de más formatos de entrada y salida (WebP, AVIF, TIFF).

## v0.5 — Más motores

- 🟡 Evaluar **CodeFormer** y **GPEN-BFR** para restauración (recorte más amplio, algo de pelo).
- 🟢 Modelos específicos para anime, texto escaneado y fotos antiguas, seleccionables desde el modo fácil.
- 🟢 Modelos de difusión (SUPIR, DiffBIR) como modo opcional para GPU con 12 GB o más: muy buenos con piel y pelo, pero tardan minutos por imagen.
- 🟢 Reducir tamaño de la descarga (cuantizar modelos ONNX, descargar los de rostros bajo demanda).

## v1.0 — Versión estable

- Criterios: pruebas manuales superadas en NVIDIA, AMD e Intel; instalador y ejecutable firmados; licencias revisadas y documentadas; sin errores conocidos abiertos; documentación de usuario y de contribución.
- 🟢 CI en GitHub Actions: `fmt`, `clippy`, tests y compilación del instalador en cada *tag*.
- 🟢 Guía de contribución (`CONTRIBUTING.md`) y plantillas de *issues*.

## Ideas sin priorizar

- Versión para macOS y Linux (el núcleo no depende de Windows; hay que reemplazar el empaquetado, la detección de hardware y la ventana).
- Línea de comandos con perfiles guardados y modo servicio.
- Previsualización en vivo mientras se ajustan los parámetros.
- Plugin para editores de imagen.

## Hecho recientemente

- Retrato completo con ampliación automática, aclarado de fotos oscuras, fuerza ajustable y comparador con zoom (ronda 16.25).
- Instalador propio con presentación de funciones, barra de progreso y desinstalador (rondas 16.25–16.26).
- Auditoría de seguridad: token por ejecución, CSP, validación de subidas y rutas, proceso hijo ligado a la app.
- Temas, paletas, tema propio, fondos animados y 10 idiomas; compatibilidad de hardware y diagnóstico.
