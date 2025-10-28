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
    
    @staticmethod
    def key_to_visual(key: str) -> str:
        """Convert a stored key ("{black_hex}_{white_hex}_{side}") to a human readable board.

        Format of key expected:
            16-hex-digits '_' 16-hex-digits '_' side
        side: 'B' or 'W' (black / white to move) – only used for an info line; it does
        not affect the stone colors (●=black, ○=white) here.

        If the key is malformed, returns a short error string.
        """
        if not isinstance(key, str):
            return "<invalid key type>"
        parts = key.strip().split('_')
        if len(parts) != 3:
            return f"<bad key format: {key}>"
        black_hex, white_hex, side = parts
        if len(black_hex) != 16 or len(white_hex) != 16:
            return f"<bad hex length: {key}>"
        try:
            black = int(black_hex, 16)
            white = int(white_hex, 16)
        except ValueError:
            return f"<bad hex digits: {key}>"
        # Render board treating black as 'me' (●) and white as 'opp' (○)
        board = Converters.bb_to_visual(black, white)
        # Append side-to-move info (B/W) for clarity
        side_info = side.upper() if side else '?'  # keep original if present
        return board + f"\n(side to move: {side_info})"
        

