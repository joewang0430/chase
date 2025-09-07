import sys, os, time, random, statistics
from typing import List, Tuple

# Ensure botzone directory on path
BOTZONE_DIR = os.path.join(os.path.dirname(__file__), 'botzone')
if BOTZONE_DIR not in sys.path:
    sys.path.append(BOTZONE_DIR)

# Import both engines
import importlib
play_mod = importlib.import_module('play')       # pure python int version
play_np_mod = importlib.import_module('play_np')  # numpy version

PythonEngine = play_mod.OthelloAI
NumpyEngine = play_np_mod.OthelloAI

random.seed(42)

# --- Helpers ---

def bitboard_to_moves_list(mv_bb: int) -> List[int]:
    res = []
    bb = mv_bb
    while bb:
        lsb = bb & -bb
        pos = lsb.bit_length() - 1
        res.append(pos)
        bb ^= lsb
    return res

# Generate random midgame positions (bitboards relative to side to move: my, opp)

def generate_positions(target: int = 5000, max_game_len: int = 60) -> List[Tuple[int,int]]:
    eng = PythonEngine()
    eng.init_standard_board(1)  # black to move; my_pieces=black, opp_pieces=white
    positions: List[Tuple[int,int]] = []
    for _ in range(target):
        # Perform a few random plies from start for variability
        # Restart from initial each sample for independence
        eng.init_standard_board(1)
        my, opp = eng.my_pieces, eng.opp_pieces
        side_moves = 0
        for _ply in range(random.randint(4, 20)):
            mv_bb = eng.generate_moves_fast(my, opp)
            if mv_bb == 0:
                # pass
                mv_bb_opp = eng.generate_moves_fast(opp, my)
                if mv_bb_opp == 0:
                    break  # game over
                # just swap turn
                my, opp = opp, my
                continue
            moves = bitboard_to_moves_list(mv_bb)
            pos = random.choice(moves)
            my, opp = eng.fast_make_move(my, opp, pos)
            # swap perspective for next ply
            my, opp = opp, my
            side_moves += 1
            if side_moves >= max_game_len:
                break
        positions.append((my, opp))
    return positions

print("Generating positions...")
positions = generate_positions()
print(f"Generated {len(positions)} positions")

# --- Benchmark Functions ---

def bench_generate(engine_type: str, positions):
    if engine_type == 'py':
        eng = PythonEngine()
        func = eng.generate_moves_fast
    else:
        eng = NumpyEngine()
        func = eng.generate_moves_fast
    # Warm-up
    for my, opp in positions[:100]:
        if engine_type == 'np':
            import numpy as np
            func(np.uint64(my), np.uint64(opp))
        else:
            func(my, opp)
    start = time.perf_counter()
    for my, opp in positions:
        if engine_type == 'np':
            import numpy as np
            func(np.uint64(my), np.uint64(opp))
        else:
            func(my, opp)
    end = time.perf_counter()
    return end - start

def bench_make_move(engine_type: str, positions):
    if engine_type == 'py':
        eng = PythonEngine()
    else:
        eng = NumpyEngine()
    times = []
    import numpy as np
    for my, opp in positions:
        # get moves
        if engine_type == 'np':
            mv_bb = eng.generate_moves_fast(np.uint64(my), np.uint64(opp))
            moves_int = int(mv_bb)
        else:
            mv_bb = eng.generate_moves_fast(my, opp)
            moves_int = mv_bb
        if moves_int == 0:
            continue
        lsb = moves_int & -moves_int
        pos = lsb.bit_length() - 1
        t0 = time.perf_counter_ns()
        if engine_type == 'np':
            eng.fast_make_move(np.uint64(my), np.uint64(opp), pos)
        else:
            eng.fast_make_move(my, opp, pos)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    if not times:
        return 0.0, 0.0, 0.0
    return statistics.mean(times), statistics.median(times), max(times)

print("Benchmarking generate_moves_fast ...")
py_time = bench_generate('py', positions)
np_time = bench_generate('np', positions)
print(f"generate_moves_fast: PurePython {py_time:.4f}s  |  NumPy {np_time:.4f}s  (positions={len(positions)})")

print("Benchmarking fast_make_move (single move timing ns)...")
py_mean, py_med, py_max = bench_make_move('py', positions)
np_mean, np_med, np_max = bench_make_move('np', positions)
print(f"fast_make_move ns (mean/median/max):")
print(f"  PurePython: mean={py_mean:.1f}  median={py_med:.1f}  max={py_max:.1f}")
print(f"  NumPy     : mean={np_mean:.1f}  median={np_med:.1f}  max={np_max:.1f}")

# Simple relative ratios
if np_time > 0:
    print(f"Speed ratio generate (NumPy / PurePython) = {np_time/py_time:.2f}x")
if np_mean > 0:
    print(f"Speed ratio make_move mean (NumPy / PurePython) = {np_mean/py_mean:.2f}x")
