"""Compara torch vs ONNX Runtime (CPU y DirectML) para las tres redes exportadas."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "pydeps"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import onnxruntime as ort  # noqa: E402
import restore_face  # noqa: E402

restore_face._patch_torchvision()
M = os.path.join(HERE, "models")
print("onnxruntime", ort.__version__, "proveedores:", ort.get_available_providers(), flush=True)


def sess(name, provider):
    prov = ["DmlExecutionProvider"] if provider == "dml" else ["CPUExecutionProvider"]
    return ort.InferenceSession(os.path.join(M, name), providers=prov)


def bench(fn, n=3):
    fn()  # calentamiento
    t = time.time()
    for _ in range(n):
        out = fn()
    return out, (time.time() - t) / n


rng = np.random.default_rng(0)


def report(label, ref, got):
    d = np.abs(ref - got)
    print(f"    {label}: max|dif|={d.max():.4f}  media={d.mean():.5f}", flush=True)


# ---------------------------------------------------------------------------- GFPGAN
print("GFPGAN v1.4 (512x512)", flush=True)
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean  # noqa: E402

net = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
                    fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)
ck = torch.load(os.path.join(ROOT, "models", "GFPGANv1.4.pth"), map_location="cpu")
net.load_state_dict(ck.get("params_ema", ck), strict=True)
net.eval()
# entrada realista: la imagen de prueba reescalada a 512 y normalizada a [-1, 1]
import cv2  # noqa: E402
img = cv2.imread(os.path.join(ROOT, "debug_uploads", "images (1).jpeg"))
img = cv2.cvtColor(cv2.resize(img, (512, 512)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
x = ((img - 0.5) / 0.5).transpose(2, 0, 1)[None].copy()

with torch.no_grad():
    ref, t_torch = bench(lambda: net(torch.from_numpy(x), return_rgb=False, randomize_noise=False)[0].numpy(), 2)
print(f"    torch CPU: {t_torch:.2f} s", flush=True)
for prov in ("cpu", "dml"):
    s = sess("gfpgan_v14.onnx", prov)
    out, t = bench(lambda: s.run(None, {"input": x})[0], 2)
    print(f"    ONNX {prov.upper()}: {t:.2f} s", flush=True)
    report(f"vs torch ({prov})", ref, out)

# ------------------------------------------------------------------------- RetinaFace
print("RetinaFace", flush=True)
from facexlib.detection import init_detection_model  # noqa: E402

det = init_detection_model("retinaface_r50" if False else "retinaface_resnet50", half=False, device="cpu",
                           model_rootpath=os.path.join(ROOT, "gfpgan", "weights"))
det.eval()
for (h, w) in [(640, 640), (480, 720)]:
    xi = rng.standard_normal((1, 3, h, w)).astype(np.float32)
    with torch.no_grad():
        ref = [t.numpy() for t in det(torch.from_numpy(xi))]
    for prov in ("cpu", "dml"):
        s = sess("retinaface_r50.onnx", prov)
        out = s.run(None, {"input": xi})
        print(f"    {w}x{h} {prov.upper()}: salidas {[o.shape for o in out]}", flush=True)
        for nm, r, o in zip(("loc", "conf", "ldm"), ref, out):
            report(nm, r, o)

# --------------------------------------------------------------------------- ParseNet
print("ParseNet", flush=True)
from facexlib.parsing import init_parsing_model  # noqa: E402

pn = init_parsing_model(model_name="parsenet", device="cpu", model_rootpath=os.path.join(ROOT, "gfpgan", "weights"))
pn.eval()
xi = rng.standard_normal((1, 3, 512, 512)).astype(np.float32)
with torch.no_grad():
    ref = pn(torch.from_numpy(xi))[0].numpy()
for prov in ("cpu", "dml"):
    s = sess("parsenet.onnx", prov)
    out = s.run(None, {"input": xi})[0]
    report(prov, ref, out)
    print(f"    etiquetas iguales: {(ref.argmax(1) == out.argmax(1)).mean() * 100:.2f}%", flush=True)
