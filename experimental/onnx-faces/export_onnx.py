"""
Exporta a ONNX las tres redes que usa el puente de rostros actual:

  1. GFPGAN v1.4            (restauración)
  2. RetinaFace ResNet50    (detección de caras + 5 puntos)
  3. ParseNet               (máscara de la cara para el pegado)

Uso (desde la raíz del proyecto, con el venv actual de torch):

    venv\\Scripts\\python.exe experimental\\onnx-faces\\export_onnx.py

Solo se usa una vez en desarrollo; el usuario final nunca necesita torch.
Las dependencias de exportación (onnx, onnxruntime-directml) viven en
experimental/onnx-faces/pydeps, sin tocar el venv principal.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "pydeps"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402
import restore_face  # noqa: E402

restore_face._patch_torchvision()

OUT = os.path.join(HERE, "models")
os.makedirs(OUT, exist_ok=True)
OPSET = 17


def export(model, dummy, name, input_names, output_names, dynamic_axes=None):
    path = os.path.join(OUT, name)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, path,
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=OPSET,
            dynamo=False, do_constant_folding=True,
        )
    print(f"  -> {name}: {os.path.getsize(path) / 1048576:.1f} MB", flush=True)
    return path


# --------------------------------------------------------------------- GFPGAN
def export_gfpgan(weights):
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

    print("GFPGAN v1.4", flush=True)
    net = GFPGANv1Clean(
        out_size=512, num_style_feat=512, channel_multiplier=2,
        decoder_load_path=None, fix_decoder=False, num_mlp=8,
        input_is_latent=True, different_w=True, narrow=1, sft_half=True,
    )
    ckpt = torch.load(weights, map_location="cpu")
    net.load_state_dict(ckpt.get("params_ema", ckpt), strict=True)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # randomize_noise=False -> ruido fijo (salida determinista)
            return self.m(x, return_rgb=False, randomize_noise=False)[0]

    export(Wrap(net), torch.randn(1, 3, 512, 512), "gfpgan_v14.onnx", ["input"], ["output"])


# ----------------------------------------------------------------- RetinaFace
def export_retinaface():
    from facexlib.detection import init_detection_model

    print("RetinaFace ResNet50", flush=True)
    det = init_detection_model("retinaface_resnet50", half=False, device="cpu",
                               model_rootpath=os.path.join(ROOT, "gfpgan", "weights"))

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            loc, conf, ldm = self.m(x)
            return loc, conf, ldm

    export(Wrap(det), torch.randn(1, 3, 640, 640), "retinaface_r50.onnx", ["input"],
           ["loc", "conf", "landmarks"],
           dynamic_axes={"input": {2: "h", 3: "w"}, "loc": {1: "n"}, "conf": {1: "n"}, "landmarks": {1: "n"}})


# -------------------------------------------------------------------- ParseNet
def export_parsenet():
    from facexlib.parsing import init_parsing_model

    print("ParseNet", flush=True)
    net = init_parsing_model(model_name="parsenet", device="cpu",
                             model_rootpath=os.path.join(ROOT, "gfpgan", "weights"))

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    export(Wrap(net), torch.randn(1, 3, 512, 512), "parsenet.onnx", ["input"], ["labels"])


# --------------------------------------------------------------- RestoreFormer++
def export_restoreformer(weights):
    sys.path.insert(0, os.path.join(ROOT, "models"))
    from RestoreFormer.modules.vqvae.vqvae_arch import VQVAEGANMultiHeadTransformer

    print("RestoreFormer++", flush=True)
    net = VQVAEGANMultiHeadTransformer(head_size=4, ex_multi_scale_num=1)
    sd = torch.load(weights, map_location="cpu")["state_dict"]
    sd = {(k[len("vqvae."):] if k.startswith("vqvae.") else k): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"  pesos: faltan {len(missing)}, sobran {len(unexpected)}", flush=True)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    export(Wrap(net), torch.randn(1, 3, 512, 512), "restoreformer_pp.onnx", ["input"], ["output"])


if __name__ == "__main__":
    which = sys.argv[1:] or ["gfpgan", "retinaface", "parsenet"]
    if "gfpgan" in which:
        export_gfpgan(os.path.join(ROOT, "models", "GFPGANv1.4.pth"))
    if "retinaface" in which:
        export_retinaface()
    if "parsenet" in which:
        export_parsenet()
    if "restoreformer" in which:
        export_restoreformer(os.path.join(ROOT, "models", "RestoreFormer++.ckpt"))
    print("listo", flush=True)
