'''
本地先生成并打印 1–6 步；
启动/前置 Othello Sensei，点击“回到开局”，然后把 1–6 步依次“点到 UI 上”（确保 UI 完全同步）；
调用新版 opening_moves_7_to_8：
50% 概率走 C 噪声（立刻点击）；
50% 概率走 oracle：等待 2 秒分析→OCR 读取→选择 top-1 最佳黄格→点击；
打印新落的 7–8 步和日志（含 oracle 的 net_win）。
'''


import random
from auto.generate_dataset import opening_moves_1_to_6, opening_moves_7_to_8
from auto.drivers import create_driver

# 1) 先生成 1-6 手（本地）
rng = random.Random(1234)
moves_1_6, black, white, side = opening_moves_1_to_6(rng=rng)
print("[LOCAL] moves_1_6:", moves_1_6, "side:", side)

# 2) 驱动 Sensei：确保前台 + 回到开局 + 点击 1-6 手，保持 UI 同步
drv = create_driver(engine_time=2.0, driver="mac", mock=False)
drv.ensure_running()
drv.reset_board()
drv.probe_click(moves_1_6, delay=0.12)
print("[CLICK] 已点击 1-6 手，准备读取 7-8 手（oracle=2s）")

# 3) 生成并点击 7-8 手（50% 噪声、50% oracle top-1；都点击）
new_moves, b2, w2, side2, logs = opening_moves_7_to_8(
    moves_so_far=list(moves_1_6),
    black_bb=black,
    white_bb=white,
    side_to_move=side,
    driver_name="mac",
    mock=False,
    rng=rng,
)
print("[RESULT] moves_7_8:", new_moves, "side:", side2)
print("[RESULT] logs:", logs)
