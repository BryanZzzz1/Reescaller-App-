<br>
<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/CustomTkinter-darkgreen?style=for-the-badge" alt="CustomTkinter">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
</div>

<br>

<div align="center">
  <h1>Herramienta de Mejora Visual 🖼️</h1>
  <p>Una aplicación de escritorio para mejorar la calidad de imágenes y videos usando IA.</p>
</div>

<br>

---

## 📖 Tabla de Contenidos

* [Acerca del Proyecto](#-acerca-del-proyecto)
* [Características Principales](#-características-principales)
* [Requisitos](#️-requisitos)
    * [Software](#software)
    * [Archivos de Modelo](#archivos-de-modelo)
* [Instalación y Uso](#️-instalación-y-uso)
    * [1. Preparar el Entorno](#1-preparar-el-entorno)
    * [2. Instalar Dependencias de Python](#2-instalar-dependencias-de-python)
    * [3. Ejecutar la Aplicación](#3-ejecutar-la-aplicación)
    * [4. Configuración Inicial](#4-configuración-inicial)
* [Contacto](#-contacto)

---

## 🚀 Acerca del Proyecto

La **Herramienta de Mejora Visual** es una aplicación de escritorio con una interfaz gráfica (GUI) que simplifica el proceso de mejorar la calidad de imágenes y videos. La aplicación automatiza el uso de potentes modelos de inteligencia artificial como **Real-ESRGAN**, **GFPGAN** y **RestoreFormer** para ofrecer una solución completa, rápida y fácil de usar para mejorar el contenido visual.

---

## ✨ Características Principales

* **Mejora de Imagen (por lote o individual):** Escala la resolución y mejora la calidad de imágenes con **Real-ESRGAN**.
* **Restauración de Rostros:** Corrige y mejora los rostros con modelos de IA de última generación (**GFPGAN** y **RestoreFormer**).
* **Mejora de Video:** Procesa videos completos fotograma a fotograma para una mejora integral, volviendo a fusionar los fotogramas mejorados con el audio original.
* **Herramientas de Video Adicionales:** Funciones integradas para convertir y recortar videos fácilmente.
* **Interfaz Intuitiva:** Interfaz de usuario limpia y moderna construida con **CustomTkinter**.

<br>
## 📸 Capturas de Pantalla

| Pantalla Principal | Pestaña de Configuración |
| :-----------------: | :--------------------: |
| <img src="https://github.com/user-attachments/assets/50967970-f747-41ee-82bf-226e293dc7c3" width="100%" alt="Captura de la pantalla principal" /> | <img src="https://github.com/user-attachments/assets/d4394aca-e73e-4c62-82b2-f87e6f28c589" width="100%" alt="Captura de la pestaña de configuración" /> |

_Interfaz principal y configuración de la aplicación._

<br>


## ⚙️ Requisitos

Asegúrate de tener el siguiente software instalado y los archivos necesarios en la carpeta principal de tu proyecto.

### Software

* **Python 3.10+**: La aplicación está desarrollada en Python 3.10 o versiones posteriores.
* **FFmpeg**: Necesario para todas las operaciones de video (extracción de fotogramas, conversión, etc.). El ejecutable `ffmpeg.exe` debe estar en la carpeta raíz del proyecto o su ruta debe configurarse manualmente.
* **Real-ESRGAN-ncnn-vulkan**: El motor principal para la mejora de imágenes. El ejecutable `realesrgan-ncnn-vulkan.exe` también debe estar en la carpeta raíz.

### Archivos de Modelo

Los modelos de IA para la restauración de rostros son necesarios para que las funciones correspondientes operen. Deben estar ubicados en la carpeta `models/`.
* `GFPGANv1.4.pth`
* `RestoreFormer++.ckpt`

---

## 🛠️ Instalación y Uso

### 1. Preparar el Entorno

Se recomienda encarecidamente usar un **entorno virtual** para evitar conflictos de dependencias.
```bash
# Crear el entorno virtual
python -m venv venv

# Activar el entorno virtual
# En Windows
venv\Scripts\activate
# En macOS/Linux
source venv/bin/activate
```
2. Instalar Dependencias de Python

Una vez que el entorno virtual esté activo, instala todas las bibliotecas de Python necesarias con pip.
```bash
pip install customtkinter Pillow opencv-python numpy
pip install torch torchvision torchaudio
pip install basicsr gfpgan
```
---
3. Ejecutar la Aplicación

Con los ejecutables de FFmpeg y Real-ESRGAN en la carpeta correcta y las dependencias instaladas, puedes iniciar la aplicación:
```bash
python All.py
```
---
🤝 Contacto

Si tienes alguna pregunta, sugerencia o encuentras algún problema, por favor, abre un issue en este repositorio.


