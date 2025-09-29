import os
import torch
from nn.model_def import Net, INPUT_PLANES

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, 'botzone', 'data')
    os.makedirs(out_dir, exist_ok=True)

    model_names = [
        'chase1_compat.pt',
        'chase2_compat.pt',
        'chase3_compat.pt',
        'chase4_compat.pt',
    ]
    # Use different seeds for reproducibility difference
    for i, name in enumerate(model_names, start=1):
        torch.manual_seed(i)
        net = Net()  # 64ch x 7 blocks
        state = net.state_dict()
        arch = {
            'channels': 64,
            'n_blocks': 7,
            'input_planes': INPUT_PLANES,
            'variant': 'net64x7_init',
            'seed': i,
        }
        path = os.path.join(out_dir, name)
        # old torch 1.8 compatible serialization
        torch.save({'arch': arch, 'model_state': state}, path, _use_new_zipfile_serialization=False)
        print(f'Wrote {path}')

if __name__ == '__main__':
    main()
