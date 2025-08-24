Herramienta de Mejora Visual 🖼️
Herramienta de Mejora Visual es una aplicación de escritorio con una interfaz gráfica (GUI) que te permite mejorar la calidad de imágenes y videos,
así como restaurar rostros, de manera sencilla y automatizada. La aplicación utiliza una combinación de potentes herramientas de código abierto como Real-ESRGAN, GFPGAN y RestoreFormer
para obtener resultados de alta calidad.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Características Principales
Mejora de Imagen Única: Aumenta la resolución y mejora la calidad de una sola imagen.

Procesamiento por Lotes: Mejora todas las imágenes dentro de una carpeta de una sola vez.

Restauración de Rostros: Utiliza modelos avanzados (GFPGAN o RestoreFormer) para restaurar y mejorar rostros en imágenes.

Mejora de Video: Automatiza el proceso de mejorar videos extrayendo, mejorando y volviendo a fusionar los fotogramas.

Herramientas de Video Adicionales: Funciones para convertir y recortar videos.

Interfaz Gráfica: Una interfaz intuitiva y fácil de usar construida con customtkinter.

Requisitos del Sistema
Para usar esta aplicación, debes tener instaladas las siguientes dependencias.

Software
Python: Versión 3.10 o superior.

FFmpeg: Requerido para el procesamiento de video (extracción, conversión y fusión). Puedes descargar el ejecutable y guardarlo en la misma carpeta que el resto del proyecto o configurar la ruta en la aplicación.

Real-ESRGAN-ncnn-vulkan: El ejecutable principal para el escalado de imágenes. Puedes descargarlo y colocarlo en el directorio de tu proyecto.

Instalación de Dependencias de Python
Es altamente recomendable usar un entorno virtual para instalar las dependencias.

Crea un entorno virtual (opcional pero recomendado):

"python -m venv mi_entorno"

Activa el entorno virtual:

Windows: mi_entorno\Scripts\activate
macOS / Linux: source mi_entorno/bin/activate

Instala las bibliotecas necesarias:

pip install customtkinter Pillow opencv-python numpy
pip install torch torchvision torchaudio
pip install basicsr gfpgan

torch y torchvision: Para los modelos de restauración de rostros. El instalador de gfpgan también lo requiere.

Instrucciones de Uso
Configura la aplicación:

Ejecuta el script principal: python All.py.

Ve a la pestaña de "Configuración".

Haz clic en "Configurar Rutas de Ejecutables" y selecciona los archivos realesrgan-ncnn-vulkan.exe y ffmpeg.exe que hayas descargado.

Selecciona los modelos de GFPGAN y RestoreFormer que hayas descargado y colocado en la carpeta models/.

Utiliza las pestañas:

"Imagen Única": Para mejorar una sola imagen.

"Restaurar Rostros": Para mejorar los rostros en una imagen.

"Carpeta": Para procesar múltiples imágenes.

"Video": Para mejorar un video completo.

Contacto
Si tienes preguntas o problemas, no dudes en abrir un issue en este repositorio.

Este proyecto es de código abierto. ¡Las contribuciones son bienvenidas!

