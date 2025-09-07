import torch
from nn.model_def import Net, build_input_planes
from botzone.play import OthelloAI

PASS_INDEX = 64

def mask_and_softmax(logits, legal_bb):
    masked = logits.clone()
    if legal_bb != 0:
        masked[:, :64] = -1e9
        m = legal_bb
        while m:
            l = m & -m
            idx = l.bit_length() - 1
            masked[0, idx] = logits[0, idx]
            m ^= l
        masked[0, PASS_INDEX] = -1e9
    else:
        masked[0, :64] = -1e9  # only PASS
    return torch.softmax(masked, dim=1)

def build_black_pass_position():
    # Board given (row1 at top -> x=0)
    # Symbols: ●=black, ○=white, ·=empty
    black = 0
    white = 0
    def set_bit(bb, x, y):
        return bb | (1 << (x*8 + y))
    # Row 1: all white
    for y in range(8):
        white = set_bit(white, 0, y)
    # Row 2: ○ ○ ○ ● ○ ● · ·
    white = set_bit(white,1,0); white = set_bit(white,1,1); white = set_bit(white,1,2); white = set_bit(white,1,4)
    black = set_bit(black,1,3); black = set_bit(black,1,5)
    # Row 3: ○ ○ ○ ○ ● ● · ·
    white = set_bit(white,2,0); white = set_bit(white,2,1); white = set_bit(white,2,2); white = set_bit(white,2,3)
    black = set_bit(black,2,4); black = set_bit(black,2,5)
    # Row 4: ○ ○ ○ ○ ● ● · ·
    white = set_bit(white,3,0); white = set_bit(white,3,1); white = set_bit(white,3,2); white = set_bit(white,3,3)
    black = set_bit(black,3,4); black = set_bit(black,3,5)
    # Row 5: ○ ○ ● ● ● ● · ·
    white = set_bit(white,4,0); white = set_bit(white,4,1)
    black = set_bit(black,4,2); black = set_bit(black,4,3); black = set_bit(black,4,4); black = set_bit(black,4,5)
    # Row 6: ○ ● ● ● · ● · ·
    white = set_bit(white,5,0)
    black = set_bit(black,5,1); black = set_bit(black,5,2); black = set_bit(black,5,3); black = set_bit(black,5,5)
    # Row 7: ○ ● ● · · ● · ·
    white = set_bit(white,6,0)
    black = set_bit(black,6,1); black = set_bit(black,6,2); black = set_bit(black,6,5)
    # Row 8: ○ · · · · · · ·
    white = set_bit(white,7,0)
    return black, white

def main():
    ai = OthelloAI()
    black, white = build_black_pass_position()
    ai.my_pieces = black  # black to move
    ai.opp_pieces = white
    legal = ai.get_legal_moves(ai.my_pieces, ai.opp_pieces)
    print(f"legal moves bitboard = {legal:016x}")
    if legal != 0:
        print("(WARNING) Expected PASS but found legal moves")
    net = Net().eval()
    with torch.no_grad():
        x = build_input_planes(ai.my_pieces, ai.opp_pieces, legal)
        logits, value = net(x)
        probs = mask_and_softmax(logits, legal)
        pass_p = float(probs[0, PASS_INDEX])
        board_sum = float(probs[0, :64].sum())
        print(f"PASS probability={pass_p:.6f} board_sum={board_sum:.6f} value={float(value.item()):.4f}")
        # Show any square prob > 1e-6
        eps = 1e-6
        sig = []
        for idx in range(64):
            p = float(probs[0, idx])
            if p > eps:
                sig.append((idx,p))
        if sig:
            print("Non-zero board probs (unexpected if pass):", sig)
        else:
            print("All board probs ~0 (correct for pass)")

if __name__ == "__main__":
    main()
