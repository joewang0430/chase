import json, subprocess, sys, os
from typing import List, Dict

# We will import the engine class directly as reference implementation
sys.path.append(os.path.join(os.path.dirname(__file__), 'botzone'))
from play import OthelloAI

CASES: List[Dict] = []

# Case A: We are black (sentinel), a short alternating history of 2 full turns.
# Botzone coordinates (x=col, y=row)
# Sequence (our moves capitalized for clarity in comment):
# Turn0: WE play (3,2)  (responses[0])
# Opp reply: (2,2)      (requests[1])
# Turn1: WE play (4,2)  (responses[1])
# Opp reply (current request): (5,2) (requests[2])
caseA = {
    "name": "Case A Black basic multi-turn",
    "payload": {
        "requests": [ {"x":-1,"y":-1}, {"x":2,"y":2}, {"x":5,"y":2} ],
        "responses": [ {"x":3,"y":2}, {"x":4,"y":2} ],
        "data": "", "globaldata": ""
    }
}
CASES.append(caseA)

# Case B: We are white (no sentinel). Opponent (black) first move (3,2); we responded (2,3);
# Now opponent second move (2,2) is current request (index == len(responses)=1).
caseB = {
    "name": "Case B White early turn",
    "payload": {
        "requests": [ {"x":3,"y":2}, {"x":2,"y":2} ],
        "responses": [ {"x":2,"y":3} ],
        "data": "", "globaldata": ""
    }
}
CASES.append(caseB)

# Case C: Black scenario with opponent PASS earlier (inserted as {-1,-1} in middle).
# Turn0 WE move (3,2), opponent replies (2,2); Turn1 WE move (4,2); opponent had no move (pass)
# Current request is opponent pass at index len(responses)=2.
caseC = {
    "name": "Case C Black with opponent pass as current request",
    "payload": {
        "requests": [ {"x":-1,"y":-1}, {"x":2,"y":2}, {"x":-1,"y":-1} ],
        "responses": [ {"x":3,"y":2}, {"x":4,"y":2} ],
        "data": "", "globaldata": ""
    }
}
CASES.append(caseC)

# Case D: White deeper: Opp moves (3,2), we (2,3), opp (2,2), we (4,2), opp current (5,2)
caseD = {
    "name": "Case D White deeper sequence",
    "payload": {
        "requests": [ {"x":3,"y":2}, {"x":2,"y":2}, {"x":5,"y":2} ],
        "responses": [ {"x":2,"y":3}, {"x":4,"y":2} ],
        "data": "", "globaldata": ""
    }
}
CASES.append(caseD)

# Case E: Programmatic random plausible sequence for black with 3 turns; we just ensure legality.
caseE = {
    "name": "Case E Black random plausible",
    "payload": {
        "requests": [ {"x":-1,"y":-1}, {"x":2,"y":3}, {"x":5,"y":4}, {"x":2,"y":2} ],
        "responses": [ {"x":3,"y":2}, {"x":4,"y":5}, {"x":5,"y":2} ],
        "data": "", "globaldata": ""
    }
}
CASES.append(caseE)

# Append extra edge cases focusing on pass handling and mixed scenarios.
extra_cases = []
# Case F: Consecutive passes (black perspective). Engine should just output PASS again (legal-move set empty assumed for reconstruction context).
extra_cases.append({
    "name": "Case F Black consecutive passes tail",
    "payload": {
        "requests": [ {"x":-1,"y":-1}, {"x":2,"y":3}, {"x":-1,"y":-1}, {"x":-1,"y":-1} ],
        "responses": [ {"x":3,"y":2}, {"x":-1,"y":-1}, {"x":-1,"y":-1} ],
        "data": "", "globaldata": ""
    }
})
# Case G: White perspective with an opponent pass in middle.
extra_cases.append({
    "name": "Case G White with opponent mid pass",
    "payload": {
        "requests": [ {"x":3,"y":2}, {"x":-1,"y":-1}, {"x":2,"y":2} ],
        "responses": [ {"x":2,"y":3} ],
        "data": "", "globaldata": ""
    }
})
# Case H: Black perspective our earlier pass then opponent moves.
extra_cases.append({
    "name": "Case H Black our earlier pass",
    "payload": {
        "requests": [ {"x":-1,"y":-1}, {"x":2,"y":3}, {"x":5,"y":3} ],
        "responses": [ {"x":-1,"y":-1}, {"x":3,"y":2} ],
        "data": "", "globaldata": ""
    }
})
CASES.extend(extra_cases)


def run_engine(json_payload: dict) -> dict:
    proc = subprocess.run([sys.executable, 'botzone/play.py'],
                          input=json.dumps(json_payload).encode(),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_line = proc.stdout.decode().strip()
    try:
        return json.loads(out_line), proc.stderr.decode()
    except json.JSONDecodeError:
        return {"_raw": out_line}, proc.stderr.decode()


def reconstruct_and_compute_expected(json_payload: dict):
    ai = OthelloAI()
    raw = json.dumps(json_payload)
    requests, responses = ai.parse_and_convert(raw)  # swapped to internal (row,col)
    ai.get_current_board(requests, responses)
    # Compute legal moves bitboard
    moves_bb = ai.generate_moves_fast(ai.my_pieces, ai.opp_pieces)
    if moves_bb == 0:
        expected_pass = True
        expected_move = (-1, -1)
    else:
        val = int(moves_bb)
        lsb = val & -val
        pos = lsb.bit_length() - 1
        expected_move = ai.bit_to_xy(pos)  # (row,col) internal
        expected_pass = False
    # Convert expected internal back to Botzone external (x=col,y=row)
    if expected_pass:
        return {"x": -1, "y": -1}, []
    else:
        row, col = expected_move
        # internal row->y, col->x externally col, row
        return {"x": col, "y": row}, extract_moves_list(moves_bb)


def extract_moves_list(moves_bb: int):
    res = []
    bb = int(moves_bb)
    while bb:
        lsb = bb & -bb
        pos = lsb.bit_length() - 1
        row = pos // 8
        col = pos % 8
        res.append((col, row))  # external (x,y)
        bb ^= lsb
    return res


def main():
    results = []
    for case in CASES:
        expected_move, legal_moves = reconstruct_and_compute_expected(case['payload'])
        output, stderr_txt = run_engine(case['payload'])
        ok = False
        detail = ''
        if 'response' in output and isinstance(output['response'], dict):
            resp = output['response']
            if expected_move['x'] == -1:
                ok = (resp['x'] == -1 and resp['y'] == -1)
                if not ok:
                    detail = f"Expected PASS got ({resp['x']},{resp['y']})"
            else:
                ok = (resp['x'] == expected_move['x'] and resp['y'] == expected_move['y'])
                if not ok:
                    detail = f"Expected {expected_move} got {resp}; legal={legal_moves}"        
        else:
            detail = f"Malformed output: {output}"
        results.append((case['name'], ok, detail, output, stderr_txt.strip()))

    # Print summary
    passed = sum(1 for r in results if r[1])
    print("================ Botzone IO Reconstruction Tests ================")
    for name, ok, detail, output, stderr_txt in results:
        status = '✅ PASS' if ok else '❌ FAIL'
        print(f"{status} {name}")
        if not ok:
            print(f"    Detail: {detail}")
            print(f"    Output: {output}")
        if stderr_txt:
            print(f"    (stderr) {stderr_txt}")
    print(f"Summary: {passed}/{len(results)} passed")
    if passed != len(results):
        sys.exit(1)

if __name__ == '__main__':
    main()
