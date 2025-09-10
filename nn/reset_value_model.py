import os
import argparse
import torch

# Allow importing nn.model_def when running from repo root
from nn.model_def import Net


def resolve_output_path(out_path: str) -> str:
    # If path is relative, resolve it from project root (two levels up from this file)
    if not os.path.isabs(out_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        out_path = os.path.join(project_root, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Initialize a fresh untrained value-only model checkpoint.')
    ap.add_argument('--out', type=str, default=os.path.join('models', 'value_only_best.pt'),
                    help='Output checkpoint path (default: models/value_only_best.pt)')
    ap.add_argument('--channels', type=int, default=64, help='Base channels for Net (default: 64)')
    ap.add_argument('--blocks', type=int, default=6, help='Number of residual blocks (default: 6)')
    args = ap.parse_args()

    out_path = resolve_output_path(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Instantiate fresh untrained network
    net = Net(channels=args.channels, n_blocks=args.blocks).cpu().eval()

    # Save as plain state_dict for broad compatibility
    state_dict = net.state_dict()
    torch.save(state_dict, out_path)

    print(f"Wrote fresh untrained checkpoint to: {out_path}")


if __name__ == '__main__':
    main()
