import os, json, math, random, time, argparse
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from model_def import Net, build_input_planes

BOARD_MASK = (1 << 64) - 1
PASS_INDEX = 64

def popcount(x: int) -> int:
    return x.bit_count() if hasattr(int, 'bit_count') else bin(x).count('1')

class ReversiValueDataset(Dataset):
    def __init__(self, path, min_empties=9):
        self.rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                empties = r.get('empties', None)
                if empties is None:
                    # fallback 计算
                    my_bb = int(r['my_bb']); opp_bb = int(r['opp_bb'])
                    occ = (my_bb | opp_bb) & BOARD_MASK
                    empties = 64 - popcount(occ)
                if empties < min_empties:  # 只训练中前盘
                    continue
                vr = r.get('value_result', None)
                if vr is None:
                    continue
                self.rows.append(r)
        if not self.rows:
            raise RuntimeError('No samples loaded (rows=0).')

    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        return (int(r['my_bb']), int(r['opp_bb']), int(r['legal_bb']), float(r['value_result']))


def collate_fn(batch):
    # batch: list of tuples
    B = len(batch)
    x = torch.zeros((B, 3, 8, 8), dtype=torch.float32)
    targets = torch.zeros(B, dtype=torch.float32)
    for i, (my_bb, opp_bb, legal_bb, val) in enumerate(batch):
        occ = my_bb | opp_bb
        m = occ
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            r, c = divmod(idx, 8)
            if (my_bb >> idx) & 1:
                x[i, 0, r, c] = 1.0
            else:
                x[i, 1, r, c] = 1.0
            m ^= lsb
        m2 = legal_bb
        while m2:
            lsb = m2 & -m2
            idx = lsb.bit_length() - 1
            r, c = divmod(idx, 8)
            x[i, 2, r, c] = 1.0
            m2 ^= lsb
        targets[i] = val
    return x, targets


def pearsonr(a, b):
    am = a.mean(); bm = b.mean()
    av = a - am; bv = b - bm
    denom = (av.pow(2).mean().sqrt() * bv.pow(2).mean().sqrt()) + 1e-9
    return (av * bv).mean() / denom


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            _, v = model(x)
            v = v.view(-1)
            preds.append(v); targets.append(y)
    preds = torch.cat(preds); targets = torch.cat(targets)
    mse = F.mse_loss(preds, targets).item()
    mae = (preds - targets).abs().mean().item()
    sign_acc = ((preds.sign() == targets.sign()) | (targets == 0)).float().mean().item()
    baseline_mse = (targets ** 2).mean().item()
    r = pearsonr(preds, targets).item()
    return dict(mse=mse, mae=mae, sign_acc=sign_acc, baseline_mse=baseline_mse, pearson_r=r)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print('Device:', device)

    dataset = ReversiValueDataset(args.data, min_empties=args.min_empties)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=0.5)

    best_val = math.inf
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            _, v = model(x)
            v = v.view(-1)
            loss = F.mse_loss(v, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()
        train_mse = running / train_size
        val_metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} train_mse={train_mse:.4f} val_mse={val_metrics['mse']:.4f} mae={val_metrics['mae']:.4f} signAcc={val_metrics['sign_acc']:.3f} r={val_metrics['pearson_r']:.3f} base={val_metrics['baseline_mse']:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if val_metrics['mse'] < best_val:
            best_val = val_metrics['mse']
            save_path = os.path.join(args.out_dir, 'value_only_best.pt')
            torch.save({'model_state': model.state_dict(), 'val_mse': best_val, 'epoch': epoch, 'config': vars(args)}, save_path)
            print('  > Saved', save_path)

    print('Total time {:.1f}s'.format(time.time()-start))
    final = evaluate(model, val_loader, device)
    print('Final:', final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-step', type=int, default=4)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--min-empties', type=int, default=9)
    ap.add_argument('--out-dir', type=str, default='models')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    train(args)

if __name__ == '__main__':
    main()
