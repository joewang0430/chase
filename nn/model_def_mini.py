"""轻量版模型 (Mini Net)
与 `model_def.Net` 结构保持一致（输入平面/各头的 FC 层不变），
但主干通道数缩减为 64，ResidualBlock 数量为 4，用于:
  - 快速调试
  - 低资源自博弈生成初始数据
  - 对比大模型 / 小模型性能差异

输出: (policy_logits(65), value(1))
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用主模型里的输入构建与 ResidualBlock，避免重复代码
from .model_def import INPUT_PLANES, build_input_planes, ResidualBlock

MINI_CHANNELS = 64
MINI_BLOCKS = 4


class MiniNet(nn.Module):
	"""轻量版 Othello 网络 (64ch x 4 blocks)"""
	def __init__(self, channels: int = MINI_CHANNELS, n_blocks: int = MINI_BLOCKS):
		super().__init__()
		assert channels == 64, "按需求固定为64通道，如需改动请显式修改调用处"
		assert n_blocks == 4, "按需求固定为4个残差块，如需改动请显式修改调用处"
		self.channels = channels
		self.n_blocks = n_blocks

		# Stem
		self.stem = nn.Sequential(
			nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True)
		)

		# Residual trunk
		self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])

		# Policy Head: Conv(64->8) -> BN -> ReLU -> FC(512->128->65)
		self.p_c = nn.Conv2d(channels, 8, 1, bias=False)
		self.p_bn = nn.BatchNorm2d(8)
		self.p_fc1 = nn.Linear(8 * 8 * 8, 128)
		self.p_fc2 = nn.Linear(128, 65)

		# Value Head: Conv(64->4) -> BN -> ReLU -> FC(256->128->64->1)
		self.v_c = nn.Conv2d(channels, 4, 1, bias=False)
		self.v_bn = nn.BatchNorm2d(4)
		self.v_fc1 = nn.Linear(4 * 8 * 8, 128)
		self.v_fc2 = nn.Linear(128, 64)
		self.v_fc3 = nn.Linear(64, 1)

	def forward(self, x: torch.Tensor):  # x: (B,4,8,8)
		x = self.stem(x)
		for blk in self.blocks:
			x = blk(x)

		# Policy
		p = F.relu(self.p_bn(self.p_c(x)), inplace=True)
		p = p.view(p.size(0), -1)  # (B,512)
		p = F.relu(self.p_fc1(p), inplace=True)
		policy_logits = self.p_fc2(p)  # (B,65)

		# Value
		v = F.relu(self.v_bn(self.v_c(x)), inplace=True)
		v = v.view(v.size(0), -1)  # (B,256)
		v = F.relu(self.v_fc1(v), inplace=True)
		v = F.relu(self.v_fc2(v), inplace=True)
		value = torch.tanh(self.v_fc3(v))  # (B,1)

		return policy_logits, value


__all__ = [
	'MiniNet', 'build_input_planes', 'INPUT_PLANES', 'MINI_CHANNELS', 'MINI_BLOCKS'
]


if __name__ == '__main__':
	# 简单自检
	my = 0x0000000810000000
	opp = 0x0000001008000000
	legal = 0x0000102004080000
	x = build_input_planes(my, opp, legal)
	net = MiniNet().eval()
	with torch.no_grad():
		p, v = net(x)
	print('MiniNet Input:', x.shape)
	print('Policy:', p.shape, 'Value:', v.shape)
