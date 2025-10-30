# Exports original 3-phase models (2025.10.30) into models.
# python3 -m nn.export_models

from pathlib import Path
import torch

from nn.model_def_128 import Net as Net128
from nn.model_def_64 import Net as Net64

OUT = Path(__file__).resolve().parents[1] / "models"
OUT.mkdir(parents=True, exist_ok=True)

def save_sd(model, path: Path):
    model.eval()
    torch.save(model.state_dict(), path)
    print(f"[SAVE] {path} ({sum(p.numel() for p in model.parameters())} params)")

def main():
    early = Net128()   # p∈[12,25]
    mid   = Net128()   # p∈[26,39]
    late  = Net64()    # p∈[40,53]

    save_sd(early, OUT / "chase_early_128.pt")
    save_sd(mid,   OUT / "chase_mid_128.pt")
    save_sd(late,  OUT / "chase_late_64.pt")

if __name__ == "__main__":
    main()