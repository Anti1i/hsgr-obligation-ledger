"""Diagnose the CUDA/cuDNN stack on an allocated GPU node.

Prints the versions that matter for the cuDNN SDPA backend, then tries one
scaled_dot_product_attention call under each backend so we can tell whether the
failure is specific to cuDNN attention or affects the whole stack.
"""
import os
import subprocess
import sys

import torch

print("python          :", sys.version.split()[0])
print("torch           :", torch.__version__)
print("torch cuda      :", torch.version.cuda)
print("cudnn (torch)   :", torch.backends.cudnn.version())
print("device          :", torch.cuda.get_device_name(0))
print("capability      :", torch.cuda.get_device_capability(0))
print("LD_LIBRARY_PATH :", os.environ.get("LD_LIBRARY_PATH", "<unset>"))

print("\n--- installed nvidia wheels ---")
out = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
for line in out.stdout.splitlines():
    if line.startswith(("nvidia-", "torch", "triton")):
        print(" ", line)

print("\n--- which libcudnn does the process actually load ---")
try:
    with open(f"/proc/{os.getpid()}/maps") as f:
        seen = sorted({ln.split()[-1] for ln in f if "libcudnn" in ln})
    print("\n".join("  " + s for s in seen) or "  none loaded yet")
except OSError as e:
    print("  unavailable:", e)

q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
backends = {
    "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
    "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
    "mem_efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    "math": torch.nn.attention.SDPBackend.MATH,
}
print("\n--- SDPA backend probe ---")
for name, backend in backends.items():
    try:
        with torch.nn.attention.sdpa_kernel(backend):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        print(f"  {name:14s} OK")
    except Exception as e:  # noqa: BLE001 - we want the failure reason verbatim
        print(f"  {name:14s} FAIL  {type(e).__name__}: {str(e).splitlines()[0][:140]}")

print("\n--- with cuDNN SDPA disabled globally ---")
torch.backends.cuda.enable_cudnn_sdp(False)
try:
    torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    print("  default SDPA OK")
except Exception as e:  # noqa: BLE001
    print(f"  default SDPA FAIL  {type(e).__name__}: {str(e).splitlines()[0][:140]}")

print("\n--- tiny real forward pass ---")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(m)
    model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16,
                                                 device_map="cuda:0")
    enc = tok("2+2=", return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    print("  generate OK:", repr(tok.decode(gen[0], skip_special_tokens=True)))
except Exception as e:  # noqa: BLE001
    print(f"  generate FAIL  {type(e).__name__}: {str(e).splitlines()[0][:200]}")
