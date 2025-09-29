"""Re-save chase8.pt using legacy (pre-zip) serialization for PyTorch 1.8 compatibility.

Usage:
  python3 export_compat.py

Outputs:
  models/chase8_compat.pt  (dict with keys: arch, model_state) saved with
  _use_new_zipfile_serialization=False so that Botzone's older torch (1.8.x)
  can load it.

The original chase8.pt may have been saved with the new zipfile serialization
introduced in newer Torch versions (>=1.6) with default=True, which older
deploy environments sometimes reject producing a plain RuntimeError during
torch.load().

We purposely store a small dict (arch + model_state) rather than the full
nn.Module object to avoid pickling class code – the single-file bot defines
the Net class inline so state_dict loading remains stable.
"""
from __future__ import annotations
import os
import torch

ORIG = os.path.join('models', 'chase8.pt')
DEST = os.path.join('models', 'chase8_compat.pt')

def main():
    if not os.path.exists(ORIG):
        print(f"[compat] source file missing: {ORIG}")
        return
    ckpt = torch.load(ORIG, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        arch = ckpt.get('arch', {})
        state = ckpt['model_state']
    elif isinstance(ckpt, dict):
        arch = {'channels': 80, 'n_blocks': 8, 'input_planes': 4}  # fallback
        state = ckpt
    else:
        raise TypeError('Unsupported checkpoint object type: ' + type(ckpt).__name__)
    obj = {'arch': arch, 'model_state': state}
    torch.save(obj, DEST, _use_new_zipfile_serialization=False)
    size = os.path.getsize(DEST)
    print(f"[compat] wrote {DEST} ({size} bytes) with arch={arch}")

if __name__ == '__main__':
    main()
