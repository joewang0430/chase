import random
import torch

from nn.model_def import Net, build_input_planes
from botzone.play import OthelloAI

PASS_INDEX = 64

ai = OthelloAI()

FULL_MASK = (1 << 64) - 1

def random_reachable_board(max_moves=40):
    """Generate a random (my_bb, opp_bb) by playing random legal moves from the start.
    Returns tuple (my_bb, opp_bb) for the side to move, plus legal_bb for that side.
    """
    # Initialize standard board with black to move (my_color=1)
    ai.init_standard_board(1)
    cur_my = ai.my_pieces
    cur_opp = ai.opp_pieces
    side_black = True  # True if current side is black (for perspective only)
    # Random number of plies
    plies = random.randint(0, max_moves)
    passes_in_row = 0
    for _ in range(plies):
        legal = ai.get_legal_moves(cur_my, cur_opp)
        if legal == 0:
            passes_in_row += 1
            if passes_in_row == 2:
                break
            # swap perspective without move (pass)
            cur_my, cur_opp = cur_opp, cur_my
            side_black = not side_black
            continue
        passes_in_row = 0
        # pick random legal move
        # enumerate bits
        moves = []
        m = legal
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            moves.append(idx)
            m ^= lsb
        pos = random.choice(moves)
        new_my, new_opp = ai.make_move(cur_my, cur_opp, pos)
        # next player's perspective
        cur_my, cur_opp = new_opp, new_my
        side_black = not side_black
    # After simulation, cur_my/cur_opp represent the side to move.
    legal_now = ai.get_legal_moves(cur_my, cur_opp)
    return cur_my, cur_opp, legal_now

def mask_and_softmax(logits: torch.Tensor, legal_bb: int) -> torch.Tensor:
    """Apply legal move mask then softmax.
    logits shape: (1,65) (0..63 board, 64=PASS)
    If legal_bb != 0 -> only legal squares allowed; PASS masked out.
    If legal_bb == 0 -> only PASS allowed.
    """
    masked = logits.clone()
    if legal_bb != 0:
        # mask all board logits then restore legal ones
        masked[:, :64] = -1e9
        m = legal_bb
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            masked[0, idx] = logits[0, idx]
            m ^= lsb
        masked[0, PASS_INDEX] = -1e9  # cannot choose PASS when moves exist
    else:
        # only PASS allowed
        masked[0, :64] = -1e9
        # keep PASS as is
    probs = torch.softmax(masked, dim=1)
    return probs

def run_trials(n=20):
    net = Net().eval()
    ok = True
    with torch.no_grad():
        for i in range(n):
            my_bb, opp_bb, legal_bb = random_reachable_board()
            # build input planes using legal_bb (third plane) for current side
            x = build_input_planes(my_bb, opp_bb, legal_bb)
            logits, value = net(x)
            probs = mask_and_softmax(logits, legal_bb)
            total_sum = float(probs.sum())
            if legal_bb != 0:
                legal_sum = 0.0
                m = legal_bb
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    legal_sum += float(probs[0, idx])
                    m ^= lsb
                pass_prob = float(probs[0, PASS_INDEX])
                print(f"[{i:02d}] moves>0  total={total_sum:.6f} legal_sum={legal_sum:.6f} pass={pass_prob:.6f} moves_cnt={bin(legal_bb).count('1')}")
                if abs(total_sum - 1.0) > 1e-6 or abs(legal_sum - 1.0) > 1e-6 or pass_prob > 1e-6:
                    ok = False
            else:
                pass_prob = float(probs[0, PASS_INDEX])
                board_sum = float(probs[0, :64].sum())
                print(f"[{i:02d}] moves=0  total={total_sum:.6f} pass={pass_prob:.6f} board_sum={board_sum:.6f}")
                if abs(total_sum - 1.0) > 1e-6 or abs(pass_prob - 1.0) > 1e-6 or board_sum > 1e-6:
                    ok = False
    if ok:
        print("All trials passed masking + softmax checks.")
    else:
        print("Discrepancy detected in masking logic.")

if __name__ == "__main__":
    run_trials(20)
