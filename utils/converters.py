class Converters:
    @staticmethod
    def bb_to_visual(me: int, opp: int) -> str:
        '''
        Convert current bit board into visible board for visualization.
        Example output:
           a b c d e f g h
        1 · · · · · · · ·
        2 · · · · · · · ·
        3 · · ● ○ · ○ · ·
        4 · · ● ● ○ ○ ● ●
        5 · ● ● ○ ● ● ● ·
        6 · · ○ ○ ● ● ○ ○
        7 · · · · ● ● · ·
        8 · · · · · · · ·
        '''
        header = "  a b c d e f g h"
        lines = [header]
        for row in range(8):
            row_cells = []
            for col in range(8):
                idx = row * 8 + col
                bit = 1 << idx
                if me & bit:
                    row_cells.append("●")
                elif opp & bit:
                    row_cells.append("○")
                else:
                    row_cells.append("·")
            # Row numbers 1..8 (top to bottom)
            lines.append(f"{row+1} " + " ".join(row_cells))
        return "\n".join(lines)
    
    @staticmethod
    def visual_to_bb(text: str) -> tuple[int, int]:
        '''
        Reverse of bb_to_visual.
        Parse an ASCII board (optionally including the header line "a b c ...")
        and return (me_bb, opp_bb).
        Accepts cell symbols:
          ● (me), ○ (opp), · or . (empty)  -- other tokens ignored.
        Extra spaces are tolerated.
        '''
        if not text:
            return 0, 0
        me = 0
        opp = 0
        lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            s = ln.lstrip()
            # Skip header line (starts with 'a ' or 'a b c...')
            if s.startswith('a ') or s.startswith('a b'):
                continue
            # Expect something like: "1 · · · · · · · ·"
            parts = s.split()
            if not parts:
                continue
            # First token should be row number 1..8
            if not parts[0].isdigit():
                continue
            row_num = int(parts[0])
            if not (1 <= row_num <= 8):
                continue
            cells = parts[1:]
            if len(cells) < 8:
                # If line concatenated without spaces, fallback char-by-char after the number
                tail = s[s.find(parts[0]) + len(parts[0]):].strip()
                if len(tail) >= 8:
                    cells = list(tail[:8])
            # Only first 8 cells considered
            for col, sym in enumerate(cells[:8]):
                idx = (row_num - 1) * 8 + col
                bit = 1 << idx
                if sym == '●':
                    me |= bit
                elif sym == '○':
                    opp |= bit
                # '.' or '·' ignored as empty
        return me, opp
    
    @staticmethod
    def dmv_to_mv(bmv: int) -> str:
        '''
        Convert decimal move to move notation
        Example: 9 -> b2
        '''
        if not isinstance(bmv, int) or bmv < 0 or bmv > 63:
            return "--"
        row = bmv // 8       # 0..7
        col = bmv % 8        # 0..7
        col_chr = chr(ord('a') + col)
        return f"{col_chr}{row+1}"
    
    @staticmethod
    def mv_to_dmv(mv: str) -> int:
        '''
        Convert move notation like "b2" -> 9
        Any invalid input (not exactly [a-h][1-8]) returns -1.
        '''
        if not mv:
            return -1
        mv = mv.strip().lower()
        if len(mv) != 2:
            return -1
        c, r = mv[0], mv[1]
        if c < 'a' or c > 'h':
            return -1
        if r < '1' or r > '8':
            return -1
        return (ord(r) - ord('1')) * 8 + (ord(c) - ord('a'))
    
    @staticmethod
    def mvs_to_bmvs(*mvs) -> int:
        '''
        Build bit mask from 1..N move notations.
        Accepts variable arguments OR a single iterable (list/tuple/set) of moves.
        Invalid moves are ignored.
        Example: mvs_to_bmvs("a1","b2") -> 0x201
        '''
        # Allow a single iterable argument: mvs_to_bmvs(["a1","b2"])
        if len(mvs) == 1 and isinstance(mvs[0], (list, tuple, set)):
            mvs = tuple(mvs[0])

        mask = 0
        for mv in mvs:
            if not isinstance(mv, str):
                continue
            idx = Converters.mv_to_dmv(mv)
            if idx >= 0:
                mask |= (1 << idx)
        return mask

