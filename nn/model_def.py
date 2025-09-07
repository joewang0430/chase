import torch
import torch.nn as nn
import torch.nn.functional as F

# 仅 3 个输入平面：
# 0: 我方棋子（始终保证是“当前准备落子一方”的棋子）
# 1: 对方棋子
# 2: 合法着点掩码（1 表示此格当前可落子）
INPUT_PLANES = 3

# 复用的全局缓冲，少分配；单线程环境安全（Botzone 单核）。
_INPUT_BUFFER = torch.zeros((INPUT_PLANES, 8, 8), dtype=torch.float32)

def build_input_planes(my_bb: int, opp_bb: int, legal_bb: int):
    """
    将位棋盘转换成 (1,3,8,8) 张量（float32）。
    my_bb / opp_bb / legal_bb 均为 64bit 整数。
    返回的张量默认可直接传入网络（无需再 clone，如果只读）。
    """
    buf = _INPUT_BUFFER
    buf.zero_()

    # 扫描所有已占格（合并 my 与 opp 只过一遍）
    occ = my_bb | opp_bb
    while occ:
        lsb = occ & -occ    # 取最低有效位
        idx = lsb.bit_length() - 1
        r, c = divmod(idx, 8)
        if (my_bb >> idx) & 1:
            buf[0, r, c] = 1.0
        else:
            buf[1, r, c] = 1.0
        occ ^= lsb

    # 扫描合法着点
    m = legal_bb
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        r, c = divmod(idx, 8)
        buf[2, r, c] = 1.0
        m ^= lsb

    return buf.unsqueeze(0)  # (1,3,8,8)

class ResidualBlock(nn.Module):
    """
    标准残差块：Conv-BN-ReLU + Conv-BN + 跳连 + ReLU
    通道数保持不变。
    """
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = self.c1(x); y = self.b1(y); y = F.relu(y, inplace=True)
        y = self.c2(y); y = self.b2(y)
        return F.relu(x + y, inplace=True)

class Net(nn.Module):
    """
    版本A:
      Stem: 3->64
      Residual Blocks: 6 个 (64 通道)
      Policy Head: Conv(64->32)->Conv(32->2)->FC(128->65)
      Value  Head: Conv(64->32)->Conv(32->1)->FC(64->64->1, tanh)
    """
    def __init__(self, channels: int = 64, n_blocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(n_blocks))
        # --- Policy 头 ---
        self.p_c = nn.Conv2d(channels, 8, 1, bias=False)   # 1x1: 64->8
        self.p_bn = nn.BatchNorm2d(8)
        self.p_fc1 = nn.Linear(8 * 8 * 8, 128)             # 512 -> 128
        self.p_fc2 = nn.Linear(128, 65)                    # 128 -> 65
        # --- Value 头 ---
        self.v_c = nn.Conv2d(channels, 4, 1, bias=False)   # 1x1: 64->4
        self.v_bn = nn.BatchNorm2d(4)
        self.v_fc1 = nn.Linear(4 * 8 * 8, 128)             # 256 -> 128
        self.v_fc2 = nn.Linear(128, 64)                    # 128 -> 64
        self.v_fc3 = nn.Linear(64, 1)                      # 64 -> 1

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        # Policy head
        p = F.relu(self.p_bn(self.p_c(x)), inplace=True)
        p = p.view(p.size(0), -1)              # (B,512)
        p = F.relu(self.p_fc1(p), inplace=True)
        policy_logits = self.p_fc2(p)          # (B,65)
        # Value head
        v = F.relu(self.v_bn(self.v_c(x)), inplace=True)
        v = v.view(v.size(0), -1)              # (B,256)
        v = F.relu(self.v_fc1(v), inplace=True)
        v = F.relu(self.v_fc2(v), inplace=True)
        value = torch.tanh(self.v_fc3(v))      # (B,1)

        return policy_logits, value
        
        
if __name__ == "__main__":
    # 自检
    my = 0x0000000810000000
    opp = 0x0000001008000000
    legal = 0x0000102004080000
    x = build_input_planes(my, opp, legal)
    net = Net().eval()
    with torch.no_grad():
        p, v = net(x)
    print("Input:", x.shape)        # (1,3,8,8)
    print("Policy:", p.shape)       # (1,65)
    print("Value:", v.shape)        # (1,1)