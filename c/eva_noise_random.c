#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Keep midgame search depth and algorithms unchanged relative to eva_noise_3.c
#define MIDGAME_DEPTH 7

typedef uint64_t u64;

typedef struct { u64 board[2]; } Board; // board[0]=current side, board[1]=opponent

typedef struct { int pos; int score; } ScoredMove;

typedef struct { int pos; int eval; } EvalMove;

// RNG (splitmix64) with env seed support
static uint64_t splitmix64_next(void) {
    static uint64_t x = 0;
    if (x == 0) {
        const char* s = getenv("CHASE_SEED");
        uint64_t seed = s ? strtoull(s, NULL, 10) : (uint64_t)time(NULL);
        x = seed ? seed : 0x9E3779B97F4A7C15ULL;
    }
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double rng_double01(void) { // [0,1)
    return (splitmix64_next() >> 11) * (1.0 / 9007199254740992.0);
}

#define MY_PIECES(b)    ((b).board[0])
#define OPP_PIECES(b)   ((b).board[1])
#define ALL_PIECES(b)   (MY_PIECES(b) | OPP_PIECES(b))
#define EMPTY_SQUARES(b) (~ALL_PIECES(b))
#define COUNT_BITS(x)   __builtin_popcountll(x)
#define SWAP_BOARD(b) do { u64 t=(b).board[0]; (b).board[0]=(b).board[1]; (b).board[1]=t; } while(0)

// Standard opening (black: (3,4),(4,3), white: (3,3),(4,4))
#define INIT_BLACK ((1ULL<<28) | (1ULL<<35))
#define INIT_WHITE ((1ULL<<27) | (1ULL<<36))

// Prototypes
u64 generate_moves(Board board);
Board make_flips(Board board, int pos);
static inline int final_score(Board b);
static inline int middle_score(Board b);
static int midgame_search(Board board, int depth, int alpha, int beta);

// Endgame solver intentionally omitted to keep this file lightweight for sampling use.

static inline int final_score(Board b){
    return (int)COUNT_BITS(MY_PIECES(b)) - (int)COUNT_BITS(OPP_PIECES(b));
}

u64 generate_moves(Board board) {
    u64 my = MY_PIECES(board);
    u64 opp = OPP_PIECES(board);
    u64 opp_inner = opp & 0x7E7E7E7E7E7E7E7EULL;
    u64 moves = 0;
    u64 b_flip, b_opp_adj;

    // E (>>1)
    b_flip = (my >> 1) & opp_inner;
    b_flip |= (b_flip >> 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 1);
    b_flip |= (b_flip >> 2) & b_opp_adj;
    b_flip |= (b_flip >> 2) & b_opp_adj;
    moves |= (b_flip >> 1) & ~(my | opp);

    // W (<<1)
    b_flip = (my << 1) & opp_inner;
    b_flip |= (b_flip << 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 1);
    b_flip |= (b_flip << 2) & b_opp_adj;
    b_flip |= (b_flip << 2) & b_opp_adj;
    moves |= (b_flip << 1) & ~(my | opp);

    // S (>>8)
    b_flip = (my >> 8) & opp;
    b_flip |= (b_flip >> 8) & opp;
    b_opp_adj = opp & (opp >> 8);
    b_flip |= (b_flip >> 16) & b_opp_adj;
    b_flip |= (b_flip >> 16) & b_opp_adj;
    moves |= (b_flip >> 8) & ~(my | opp);

    // N (<<8)
    b_flip = (my << 8) & opp;
    b_flip |= (b_flip << 8) & opp;
    b_opp_adj = opp & (opp << 8);
    b_flip |= (b_flip << 16) & b_opp_adj;
    b_flip |= (b_flip << 16) & b_opp_adj;
    moves |= (b_flip << 8) & ~(my | opp);

    // SE (>>7)
    b_flip = (my >> 7) & opp_inner;
    b_flip |= (b_flip >> 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 7);
    b_flip |= (b_flip >> 14) & b_opp_adj;
    b_flip |= (b_flip >> 14) & b_opp_adj;
    moves |= (b_flip >> 7) & ~(my | opp);

    // NW (<<7)
    b_flip = (my << 7) & opp_inner;
    b_flip |= (b_flip << 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 7);
    b_flip |= (b_flip << 14) & b_opp_adj;
    b_flip |= (b_flip << 14) & b_opp_adj;
    moves |= (b_flip << 7) & ~(my | opp);

    // NE (>>9)
    b_flip = (my >> 9) & opp_inner;
    b_flip |= (b_flip >> 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 9);
    b_flip |= (b_flip >> 18) & b_opp_adj;
    b_flip |= (b_flip >> 18) & b_opp_adj;
    moves |= (b_flip >> 9) & ~(my | opp);

    // SW (<<9)
    b_flip = (my << 9) & opp_inner;
    b_flip |= (b_flip << 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 9);
    b_flip |= (b_flip << 18) & b_opp_adj;
    b_flip |= (b_flip << 18) & b_opp_adj;
    moves |= (b_flip << 9) & ~(my | opp);

    return moves;
}

Board make_flips(Board board, int pos) {
    u64 my = MY_PIECES(board);
    u64 opp = OPP_PIECES(board);
    u64 opp_inner = opp & 0x7E7E7E7E7E7E7E7EULL;
    u64 pos_bit = 1ULL << pos;
    u64 flipped = 0;
    u64 b_flip, b_opp_adj;

    // E (>>1)
    b_flip = (pos_bit >> 1) & opp_inner;
    b_flip |= (b_flip >> 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 1);
    b_flip |= (b_flip >> 2) & b_opp_adj;
    b_flip |= (b_flip >> 2) & b_opp_adj;
    if ((b_flip >> 1) & my) flipped |= b_flip;

    // W (<<1)
    b_flip = (pos_bit << 1) & opp_inner;
    b_flip |= (b_flip << 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 1);
    b_flip |= (b_flip << 2) & b_opp_adj;
    b_flip |= (b_flip << 2) & b_opp_adj;
    if ((b_flip << 1) & my) flipped |= b_flip;

    // S (>>8)
    b_flip = (pos_bit >> 8) & opp;
    b_flip |= (b_flip >> 8) & opp;
    b_opp_adj = opp & (opp >> 8);
    b_flip |= (b_flip >> 16) & b_opp_adj;
    b_flip |= (b_flip >> 16) & b_opp_adj;
    if ((b_flip >> 8) & my) flipped |= b_flip;

    // N (<<8)
    b_flip = (pos_bit << 8) & opp;
    b_flip |= (b_flip << 8) & opp;
    b_opp_adj = opp & (opp << 8);
    b_flip |= (b_flip << 16) & b_opp_adj;
    b_flip |= (b_flip << 16) & b_opp_adj;
    if ((b_flip << 8) & my) flipped |= b_flip;

    // SE (>>7)
    b_flip = (pos_bit >> 7) & opp_inner;
    b_flip |= (b_flip >> 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 7);
    b_flip |= (b_flip >> 14) & b_opp_adj;
    b_flip |= (b_flip >> 14) & b_opp_adj;
    if ((b_flip >> 7) & my) flipped |= b_flip;

    // NW (<<7)
    b_flip = (pos_bit << 7) & opp_inner;
    b_flip |= (b_flip << 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 7);
    b_flip |= (b_flip << 14) & b_opp_adj;
    b_flip |= (b_flip << 14) & b_opp_adj;
    if ((b_flip << 7) & my) flipped |= b_flip;

    // NE (>>9)
    b_flip = (pos_bit >> 9) & opp_inner;
    b_flip |= (b_flip >> 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 9);
    b_flip |= (b_flip >> 18) & b_opp_adj;
    b_flip |= (b_flip >> 18) & b_opp_adj;
    if ((b_flip >> 9) & my) flipped |= b_flip;

    // SW (<<9)
    b_flip = (pos_bit << 9) & opp_inner;
    b_flip |= (b_flip << 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 9);
    b_flip |= (b_flip << 18) & b_opp_adj;
    b_flip |= (b_flip << 18) & b_opp_adj;
    if ((b_flip << 9) & my) flipped |= b_flip;

    Board result;
    result.board[0] = my | pos_bit | flipped;
    result.board[1] = opp & ~flipped;
    return result;
}

static inline int middle_score(Board b) {
    // Adopt the ajex_bb_rvc.c midgame evaluation: piece count + corner control + mobility
    const u64 CORNER_MASK = 0x8100000000000081ULL;
    const int cornerW = 17;
    const int mobilityW = 10;

    int myPieces = COUNT_BITS(MY_PIECES(b));
    int oppPieces = COUNT_BITS(OPP_PIECES(b));

    u64 myCorners = MY_PIECES(b) & CORNER_MASK;
    u64 oppCorners = OPP_PIECES(b) & CORNER_MASK;
    int myCornerCnt = COUNT_BITS(myCorners);
    int oppCornerCnt = COUNT_BITS(oppCorners);

    u64 myMoves = generate_moves(b);
    Board opp = b; SWAP_BOARD(opp);
    u64 oppMoves = generate_moves(opp);
    int myMob = COUNT_BITS(myMoves);
    int oppMob = COUNT_BITS(oppMoves);

    int myScore = myPieces + cornerW * myCornerCnt + mobilityW * myMob;
    int oppScore = oppPieces + cornerW * oppCornerCnt + mobilityW * oppMob;
    return myScore - oppScore;
}

static int midgame_search(Board board, int depth, int alpha, int beta) {
    if (depth == 0) return middle_score(board);
    u64 moves = generate_moves(board);
    if (!moves) {
        Board opp = board; SWAP_BOARD(opp);
        u64 opp_moves = generate_moves(opp);
        if (!opp_moves) return final_score(board);
        return -midgame_search(opp, depth, -beta, -alpha);
    }
    ScoredMove sorted_moves[32];
    int count = 0;
    u64 t = moves;
    while (t) {
        int pos = __builtin_ctzll(t);
        t &= (t - 1);
        // simple move ordering by heuristic evaluate (PST+flips)
        Board nb = make_flips(board, pos);
        int flips = COUNT_BITS(MY_PIECES(nb)) - COUNT_BITS(MY_PIECES(board)) - 1;
        int edge = (pos < 8 || pos >= 56 || (pos % 8) == 0 || (pos % 8) == 7) ? 8 : 0;
        int corner = (pos == 0 || pos == 7 || pos == 56 || pos == 63) ? 40 : 0;
        sorted_moves[count].pos = pos;
        sorted_moves[count].score = corner + edge + flips;
        count++;
    }
    // insertion sort (count small)
    for (int i = 1; i < count; ++i) {
        ScoredMove key = sorted_moves[i];
        int j = i - 1;
        while (j >= 0 && sorted_moves[j].score < key.score) { sorted_moves[j+1] = sorted_moves[j]; --j; }
        sorted_moves[j+1] = key;
    }
    int best = -INT_MAX;
    for (int i = 0; i < count; ++i) {
        int pos = sorted_moves[i].pos;
        Board nb = make_flips(board, pos);
        SWAP_BOARD(nb);
        int val = -midgame_search(nb, depth - 1, -beta, -alpha);
        if (val > best) best = val;
        if (val > alpha) alpha = val;
        if (alpha >= beta) break;
    }
    return best;
}

// Public entry: choose a move with heavy randomness but bias against awful moves.
int choose_move(uint64_t my_pieces, uint64_t opp_pieces) {
    Board board = (Board){{my_pieces, opp_pieces}};
    u64 moves = generate_moves(board);
    if (!moves) return -1; // PASS

    // Opening symmetric random (first move)
    if (MY_PIECES(board) == INIT_BLACK && OPP_PIECES(board) == INIT_WHITE) {
        int cand[4]; int m = 0;
        u64 t = moves;
        while (t) {
            int pos = __builtin_ctzll(t);
            t &= (t - 1);
            if (m < 4) cand[m++] = pos;
        }
        if (m > 0) {
            int pick = (int)(rng_double01() * m);
            if (pick >= m) pick = m - 1;
            return cand[pick];
        }
    }

    // Evaluate all legal moves using the same midgame search (depth unchanged)
    EvalMove evals[32];
    int n = 0;
    u64 t = moves;
    while (t) {
        int pos = __builtin_ctzll(t);
        t &= (t - 1);
        Board nb = make_flips(board, pos);
        SWAP_BOARD(nb);
        int val = -midgame_search(nb, MIDGAME_DEPTH - 1, -INT_MAX, INT_MAX);
        evals[n].pos = pos;
        evals[n].eval = val;
        n++;
    }
    if (n <= 1) return (n == 1 ? evals[0].pos : -1);

    // Softmax over ALL legal moves with a BIG temperature to be almost random but still penalize awful moves
    // Temperature from env CHASE_NOISE_TEMP (default 60.0)
    double T = 60.0;
    const char* st = getenv("CHASE_NOISE_TEMP");
    if (st) {
        double tv = atof(st);
        if (tv > 1e-6) T = tv;
    }

    // Numerical stability: shift by max eval
    int max_eval = evals[0].eval;
    for (int i = 1; i < n; ++i) if (evals[i].eval > max_eval) max_eval = evals[i].eval;

    double wsum = 0.0;
    double w[32];
    for (int i = 0; i < n; ++i) {
        double s = (double)(evals[i].eval - max_eval) / T; // large T -> flatter
        // clamp s to avoid underflow; allow very negative to strongly penalize awful moves
        if (s < -50.0) s = -50.0;
        w[i] = exp(s);
        // add small uniform epsilon so absolutely terrible moves still have tiny chance
        w[i] += 1e-6;
        wsum += w[i];
    }

    if (wsum <= 0.0) return evals[(int)(rng_double01() * n) % n].pos;

    // Sample
    double u = rng_double01() * wsum;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += w[i];
        if (u <= acc) return evals[i].pos;
    }
    return evals[n - 1].pos; // fallback
}
