import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""Othello NN (128 channels, 8 residual blocks)
Inputs (4 planes, 8x8):
  0: current player's stones (my)
  1: opponent stones (opp)
  2: empty squares (empty)
  3: legal moves for current player (legal)
Outputs:
  - policy: logits over 64 board positions (no pass)
  - value: scalar in [-1, 1]
"""

INPUT_PLANES = 4

# Shared scratch buffer to avoid small allocations (single-thread safe in our setting)
_INPUT_BUFFER = torch.zeros((INPUT_PLANES, 8, 8), dtype=torch.float32)


def build_input_planes(my_bb: int, opp_bb: int, legal_bb: int) -> torch.Tensor:
    """Build input tensor of shape (1, 4, 8, 8) from bitboards.

    Args:
        my_bb:   bitboard for current player's stones
        opp_bb:  bitboard for opponent stones
        legal_bb: bitboard of legal moves for current player
    Returns:
        torch.FloatTensor with planes [my, opp, empty, legal]
    """
    buf = _INPUT_BUFFER
    buf.zero_()

    # Occupied squares (my | opp)
    occ = my_bb | opp_bb
    while occ:
        lsb = occ & -occ
        idx = lsb.bit_length() - 1
        r, c = divmod(idx, 8)
        if (my_bb >> idx) & 1:
            buf[0, r, c] = 1.0
        else:
            buf[1, r, c] = 1.0
        occ ^= lsb

    FULL_MASK = 0xFFFFFFFFFFFFFFFF
    empty_bb = (~(my_bb | opp_bb)) & FULL_MASK
    m = empty_bb
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        r, c = divmod(idx, 8)
        buf[2, r, c] = 1.0
        m ^= lsb

    m = legal_bb
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        r, c = divmod(idx, 8)
        buf[3, r, c] = 1.0
        m ^= lsb

    return buf.unsqueeze(0)


class ResidualBlock(nn.Module):
    """Standard residual block: Conv-BN-ReLU -> Conv-BN + skip -> ReLU."""

    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(x)
        y = self.b1(y)
        y = F.relu(y, inplace=True)
        y = self.c2(y)
        y = self.b2(y)
        return F.relu(x + y, inplace=True)


class Net(nn.Module):
    """8-block residual network with 128 channels.

    Policy head: 1x1 Conv(C->8) -> BN -> ReLU -> FC(512->256->64)
    Value  head: 1x1 Conv(C->4) -> BN -> ReLU -> FC(256->256->128->1) with tanh
    """

    def __init__(self, channels: int = 128, n_blocks: int = 8):
        super().__init__()
        self.channels = channels
        self.n_blocks = n_blocks

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Trunk
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])

        # Policy head
        self.p_c = nn.Conv2d(channels, 8, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(8)
        self.p_fc1 = nn.Linear(8 * 8 * 8, 256)  # 512 -> 256
        self.p_fc2 = nn.Linear(256, 64)         # 256 -> 64 logits (no pass)

        # Value head
        self.v_c = nn.Conv2d(channels, 4, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(4)
        self.v_fc1 = nn.Linear(4 * 8 * 8, 256)  # 256 -> 256
        self.v_fc2 = nn.Linear(256, 128)        # 256 -> 128
        self.v_fc3 = nn.Linear(128, 1)          # 128 -> 1

        self.tanh = nn.Tanh()
        self._init_weights()

    @staticmethod
    def _kaiming_conv_(m: nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    @staticmethod
    def _kaiming_linear_(m: nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._kaiming_conv_(m)
            elif isinstance(m, nn.Linear):
                self._kaiming_linear_(m)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)

        # Policy
        p = F.relu(self.p_bn(self.p_c(x)), inplace=True)
        p = p.view(p.size(0), -1)  # (B, 512)
        p = F.relu(self.p_fc1(p), inplace=True)
        policy_logits = self.p_fc2(p)

        # Value
        v = F.relu(self.v_bn(self.v_c(x)), inplace=True)
        v = v.view(v.size(0), -1)  # (B, 256)
        v = F.relu(self.v_fc1(v), inplace=True)
        v = F.relu(self.v_fc2(v), inplace=True)
        value = self.tanh(self.v_fc3(v))

        return policy_logits, value


if __name__ == "__main__":
    # Quick self-check
    my = 0x0000000810000000
    opp = 0x0000001008000000
    legal = 0x0000102004080000
    x = build_input_planes(my, opp, legal)
    net = Net().eval()
    with torch.no_grad():
        p, v = net(x)
    print("Input:", x.shape)      # (1,4,8,8)
    print("Policy:", p.shape)     # (1,64)
    print("Value:", v.shape)      # (1,1)
