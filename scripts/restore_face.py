#!/usr/bin/env python
"""
Puente de rostros para la app Rust (reescaller).

Los modelos GFPGAN y RestoreFormer++ siguen siendo PyTorch: la aplicación en
Rust los invoca a través de este script para no duplicar el stack de IA.

Uso:
    python restore_face.py --model gfpgan --weights models/GFPGANv1.4.pth \
                           --input cara.jpg --output cara_rostros.png

También soporta:
    python restore_face.py --selftest     # comprueba que torch/gfpgan importan
"""

import argparse
import os
import sys


def _patch_torchvision():
    """Recrea `torchvision.transforms.functional_tensor` (eliminado en torchvision >= 0.17)."""
    try:
        import types

        import torch  # noqa: F401
        import torchvision
        import torchvision.transforms.functional as F
    except Exception:
        return False

    if hasattr(torchvision.transforms, "functional_tensor"):
        return True

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    for name in dir(F):
        if not name.startswith("_"):
            try:
                setattr(shim, name, getattr(F, name))
            except Exception:
                pass
    if not hasattr(shim, "rgb_to_grayscale") and hasattr(shim, "to_grayscale"):
        shim.rgb_to_grayscale = shim.to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim
    torchvision.transforms.functional_tensor = shim
    return True


def _prepare_paths():
    """Añade `models/` al sys.path para poder importar RestoreFormer."""
    here = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(here)
    models_dir = os.path.join(project, "models")
    for path in (models_dir, project):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def restore_gfpgan(weights, input_path, output_path):
    import cv2
    import torch
    from gfpgan import GFPGANer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gfpgan] dispositivo: {device}", flush=True)
    restaurador = GFPGANer(
        model_path=weights, upscale=2, bg_upsampler=None, device=device
    )

    imagen = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if imagen is None:
        raise RuntimeError(f"No se pudo leer la imagen: {input_path}")

    _, _, salida = restaurador.enhance(
        imagen, has_aligned=False, only_center_face=False
    )
    if salida is None:
        raise RuntimeError("No se encontraron rostros o la restauración falló.")

    cv2.imwrite(output_path, salida)
    return salida.shape


def restore_restoreformer(weights, input_path, output_path):
    import cv2
    from basicsr.utils import imwrite
    from RestoreFormer.RestoreFormer import RestoreFormer

    restaurador = RestoreFormer(
        model_path=weights,
        upscale=2,
        arch="RestoreFormer++",
        bg_upsampler=None,
    )

    imagen = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if imagen is None:
        raise RuntimeError(f"No se pudo leer la imagen: {input_path}")

    _, _, salida = restaurador.enhance(
        imagen, has_aligned=False, only_center_face=False, paste_back=True
    )
    if salida is None:
        raise RuntimeError("No se encontraron rostros o la restauración falló.")

    imwrite(salida, output_path)
    return salida.shape


def main():
    parser = argparse.ArgumentParser(description="Restauración de rostros")
    parser.add_argument("--model", default="gfpgan", help="gfpgan | restoreformer")
    parser.add_argument("--weights", help="Ruta del archivo de pesos")
    parser.add_argument("--input", help="Imagen de entrada")
    parser.add_argument("--output", help="Imagen de salida")
    parser.add_argument(
        "--selftest", action="store_true", help="Solo comprueba las dependencias"
    )
    args = parser.parse_args()

    _prepare_paths()
    _patch_torchvision()

    if args.selftest:
        import torch  # noqa: F401
        import gfpgan  # noqa: F401
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        print("OK: torch + gfpgan + cv2 + numpy importan correctamente")
        return 0

    if not args.weights or not args.input or not args.output:
        parser.error("faltan --weights, --input o --output")
    if not os.path.isfile(args.weights):
        raise SystemExit(f"No se encontró el modelo: {args.weights}")
    if not os.path.isfile(args.input):
        raise SystemExit(f"No se encontró la imagen: {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    model = args.model.lower()
    if model in ("gfpgan", "gfpganv1.4"):
        forma = restore_gfpgan(args.weights, args.input, args.output)
    elif model in ("restoreformer", "restoreformer++", "rf++"):
        forma = restore_restoreformer(args.weights, args.input, args.output)
    else:
        raise SystemExit(f"Modelo desconocido: {args.model}")

    print(f"[{model}] salida guardada en {args.output} ({forma[1]}x{forma[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
