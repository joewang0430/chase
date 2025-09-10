'''
reset_chase.py: 初始化未训练网络并保存为 state_dict。
参数：
--out 默认 chase.pt
--channels 默认 64
--blocks 默认 6
--seed 可选，固定初始化以便复现
'''

# 运行：
# PYTHONPATH=. python3 nn/reset_chase.py --out models/chase.pt --seed 42

import os
import argparse
import torch

from nn.model_def import Net


def resolve_output_path(out_path: str) -> str:
    if not os.path.isabs(out_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        out_path = os.path.join(project_root, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Initialize a fresh untrained Net and save to models/chase.pt')
    parser.add_argument('--out', type=str, default=os.path.join('models', 'chase.pt'),
                        help='Output checkpoint path (default: models/chase.pt)')
    parser.add_argument('--channels', type=int, default=64, help='Base channels for Net (default: 64)')
    parser.add_argument('--blocks', type=int, default=6, help='Number of residual blocks (default: 6)')
    parser.add_argument('--seed', type=int, default=None, help='Optional seed for deterministic init')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    out_path = resolve_output_path(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    net = Net(channels=args.channels, n_blocks=args.blocks).cpu().eval()
    torch.save(net.state_dict(), out_path)
    print(f"Wrote fresh untrained checkpoint to: {out_path}")


if __name__ == '__main__':
    main()
