#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>

#define MIDGAME_DEPTH 8     // 8
#define ENDING_DEPTH 16     // 16

typedef uint64_t u64;

typedef struct { u64 board[2]; } Board; // board[0]=当前方, board[1]=对方

typedef struct {
    int pos;
    int score;
} ScoredMove;

#define MY_PIECES(b)    ((b).board[0])
#define OPP_PIECES(b)   ((b).board[1])
#define ALL_PIECES(b)   (MY_PIECES(b) | OPP_PIECES(b))
#define EMPTY_SQUARES(b) (~ALL_PIECES(b))
#define COUNT_BITS(x)   __builtin_popcountll(x)
#define MY_COUNT(b)     COUNT_BITS(MY_PIECES(b))
#define OPP_COUNT(b)    COUNT_BITS(OPP_PIECES(b))
#define SWAP_BOARD(b) do { u64 t=(b).board[0]; (b).board[0]=(b).board[1]; (b).board[1]=t; } while(0)

int choose_move(uint64_t my_pieces, uint64_t opp_pieces);

// 基础函数
u64 generate_moves(Board board);
Board make_flips(Board board, int pos);
int solve_endgame(uint64_t my_pieces, uint64_t opp_pieces, int* best_move);
static inline int final_score(Board b);
static inline int middle_score(Board b);

// 排序移动相关函数
static int evaluate_move(Board board, int pos);
static int compare_moves(const void *a, const void *b);
static int get_sorted_moves(Board board, u64 moves, ScoredMove *sorted_moves);

// pvs优化后
static int perfect_search_pvs(Board board, int empties, int alpha, int beta, bool is_pv);

// 结束后棋盘打分
static inline int final_score(Board b){ 
    return (int)MY_COUNT(b) - (int)OPP_COUNT(b); 
} 

// 完全枚举搜索 (Negamax + AlphaBeta)，同时pvs优化
static int perfect_search_pvs(Board board, int empties, int alpha, int beta, bool is_pv){
    u64 moves = generate_moves(board);
    if (!moves){
        Board opp = board; SWAP_BOARD(opp);
        u64 opp_moves = generate_moves(opp);
        if (!opp_moves){ // 双方都无棋 -> 终局
            return final_score(board);
        }
        // 单方 PASS，不减少剩余空位(棋盘未变化)，换边取负
        int val = -perfect_search_pvs(opp, empties, -beta, -alpha, is_pv);
        return val;
    }
    if (empties == 0){ // 理论上不应发生(有合法着说明仍有空格) 保险返回
        return final_score(board);
    }
    
    int best = INT_MIN;
    ScoredMove sorted_moves[32];
    int move_count = get_sorted_moves(board, moves, sorted_moves);
    
    bool first_move = true;
    for (int i = 0; i < move_count; i++) {
        int pos = sorted_moves[i].pos;
        Board nb = make_flips(board, pos);
        SWAP_BOARD(nb);
        
        int val;
        if (first_move) {
            // 第一个移动：用全窗口搜索（这很可能是最佳移动）
            val = -perfect_search_pvs(nb, empties - 1, -beta, -alpha, true);
            first_move = false;
        } else {
            // 后续移动：先用null window试探
            val = -perfect_search_pvs(nb, empties - 1, -alpha - 1, -alpha, false);
            
            // 如果试探发现这个移动很好，重新用全窗口搜索
            if (val > alpha && val < beta && is_pv) {
                val = -perfect_search_pvs(nb, empties - 1, -beta, -alpha, true);
            }
        }
        if (val > best) best = val;
        if (val > alpha) alpha = val;
        if (alpha >= beta) break; // 剪枝
    }
    return best;
}

int solve_endgame(uint64_t my_pieces, uint64_t opp_pieces, int* best_move){
    Board board = {{my_pieces, opp_pieces}};
    int empties = 64 - COUNT_BITS(my_pieces | opp_pieces);

    u64 moves = generate_moves(board);
    if (!moves){
        Board opp = board; SWAP_BOARD(opp);
        u64 opp_moves = generate_moves(opp);
        if (!opp_moves){ *best_move = -1; return final_score(board); }
        int score = -perfect_search_pvs(opp, empties, -INT_MAX, INT_MAX, true);
        *best_move = -1; return score;
    }

    int alpha = -INT_MAX, beta = INT_MAX;
    int best_score = INT_MIN; int best_pos = -1;
    u64 tmp = moves;
    // 使用排序后的移动
    ScoredMove sorted_moves[32];
    int move_count = get_sorted_moves(board, moves, sorted_moves);

    for (int i = 0; i < move_count; i++) {
        int pos = sorted_moves[i].pos;
        Board nb = make_flips(board, pos);
        SWAP_BOARD(nb);
        int score = -perfect_search_pvs(nb, empties - 1, -beta, -alpha, true); // 使用PVS
        if (score > best_score){ best_score = score; best_pos = pos; }
        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }
    *best_move = best_pos;
    return best_score;
}

u64 generate_moves(Board board) {
    u64 my = MY_PIECES(board);
    u64 opp = OPP_PIECES(board);
    u64 opp_inner = opp & 0x7E7E7E7E7E7E7E7EULL; // 排除边界，用于水平/对角移动
    u64 moves = 0;
    u64 b_flip, b_opp_adj;
    
    // 水平向右 (>>1)
    b_flip = (my >> 1) & opp_inner;
    b_flip |= (b_flip >> 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 1);
    b_flip |= (b_flip >> 2) & b_opp_adj;
    b_flip |= (b_flip >> 2) & b_opp_adj;
    moves |= (b_flip >> 1) & ~(my | opp);
    
    // 水平向左 (<<1)
    b_flip = (my << 1) & opp_inner;
    b_flip |= (b_flip << 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 1);
    b_flip |= (b_flip << 2) & b_opp_adj;
    b_flip |= (b_flip << 2) & b_opp_adj;
    moves |= (b_flip << 1) & ~(my | opp);
    
    // 垂直向下 (>>8)
    b_flip = (my >> 8) & opp;
    b_flip |= (b_flip >> 8) & opp;
    b_opp_adj = opp & (opp >> 8);
    b_flip |= (b_flip >> 16) & b_opp_adj;
    b_flip |= (b_flip >> 16) & b_opp_adj;
    moves |= (b_flip >> 8) & ~(my | opp);
    
    // 垂直向上 (<<8)
    b_flip = (my << 8) & opp;
    b_flip |= (b_flip << 8) & opp;
    b_opp_adj = opp & (opp << 8);
    b_flip |= (b_flip << 16) & b_opp_adj;
    b_flip |= (b_flip << 16) & b_opp_adj;
    moves |= (b_flip << 8) & ~(my | opp);
    
    // 对角线右下 (>>7)
    b_flip = (my >> 7) & opp_inner;
    b_flip |= (b_flip >> 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 7);
    b_flip |= (b_flip >> 14) & b_opp_adj;
    b_flip |= (b_flip >> 14) & b_opp_adj;
    moves |= (b_flip >> 7) & ~(my | opp);
    
    // 对角线左上 (<<7)
    b_flip = (my << 7) & opp_inner;
    b_flip |= (b_flip << 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 7);
    b_flip |= (b_flip << 14) & b_opp_adj;
    b_flip |= (b_flip << 14) & b_opp_adj;
    moves |= (b_flip << 7) & ~(my | opp);
    
    // 对角线右上 (>>9)
    b_flip = (my >> 9) & opp_inner;
    b_flip |= (b_flip >> 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 9);
    b_flip |= (b_flip >> 18) & b_opp_adj;
    b_flip |= (b_flip >> 18) & b_opp_adj;
    moves |= (b_flip >> 9) & ~(my | opp);
    
    // 对角线左下 (<<9)
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
    
    // 水平向右 (>>1)
    b_flip = (pos_bit >> 1) & opp_inner;
    b_flip |= (b_flip >> 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 1);
    b_flip |= (b_flip >> 2) & b_opp_adj;
    b_flip |= (b_flip >> 2) & b_opp_adj;
    if ((b_flip >> 1) & my) flipped |= b_flip;
    
    // 水平向左 (<<1)
    b_flip = (pos_bit << 1) & opp_inner;
    b_flip |= (b_flip << 1) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 1);
    b_flip |= (b_flip << 2) & b_opp_adj;
    b_flip |= (b_flip << 2) & b_opp_adj;
    if ((b_flip << 1) & my) flipped |= b_flip;
    
    // 垂直向下 (>>8)
    b_flip = (pos_bit >> 8) & opp;
    b_flip |= (b_flip >> 8) & opp;
    b_opp_adj = opp & (opp >> 8);
    b_flip |= (b_flip >> 16) & b_opp_adj;
    b_flip |= (b_flip >> 16) & b_opp_adj;
    if ((b_flip >> 8) & my) flipped |= b_flip;
    
    // 垂直向上 (<<8)
    b_flip = (pos_bit << 8) & opp;
    b_flip |= (b_flip << 8) & opp;
    b_opp_adj = opp & (opp << 8);
    b_flip |= (b_flip << 16) & b_opp_adj;
    b_flip |= (b_flip << 16) & b_opp_adj;
    if ((b_flip << 8) & my) flipped |= b_flip;
    
    // 对角线右下 (>>7)
    b_flip = (pos_bit >> 7) & opp_inner;
    b_flip |= (b_flip >> 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 7);
    b_flip |= (b_flip >> 14) & b_opp_adj;
    b_flip |= (b_flip >> 14) & b_opp_adj;
    if ((b_flip >> 7) & my) flipped |= b_flip;
    
    // 对角线左上 (<<7)
    b_flip = (pos_bit << 7) & opp_inner;
    b_flip |= (b_flip << 7) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner << 7);
    b_flip |= (b_flip << 14) & b_opp_adj;
    b_flip |= (b_flip << 14) & b_opp_adj;
    if ((b_flip << 7) & my) flipped |= b_flip;
    
    // 对角线右上 (>>9)
    b_flip = (pos_bit >> 9) & opp_inner;
    b_flip |= (b_flip >> 9) & opp_inner;
    b_opp_adj = opp_inner & (opp_inner >> 9);
    b_flip |= (b_flip >> 18) & b_opp_adj;
    b_flip |= (b_flip >> 18) & b_opp_adj;
    if ((b_flip >> 9) & my) flipped |= b_flip;
    
    // 对角线左下 (<<9)
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

// 移动评估函数
static int evaluate_move(Board board, int pos) {
    // 1. 角落权重最高
    if (pos == 0 || pos == 7 || pos == 56 || pos == 63) {
        return 1000;
    }
    // 2. 边缘次优
    if (pos < 8 || pos >= 56 || (pos % 8) == 0 || (pos % 8) == 7) {
        return 100;
    }
    // 3. 翻转子数量
    u64 my = MY_PIECES(board);
    u64 opp = OPP_PIECES(board);
    Board test_board = make_flips(board, pos);
    int flips = COUNT_BITS(MY_PIECES(test_board)) - COUNT_BITS(my) - 1;
    return flips * 10;
}

// 比较函数：分数高的排在前面
static int compare_moves(const void *a, const void *b) {
    const ScoredMove *ma = (const ScoredMove*)a;
    const ScoredMove *mb = (const ScoredMove*)b;
    return mb->score - ma->score;  // 降序排列
}

static int get_sorted_moves(Board board, u64 moves, ScoredMove *sorted_moves) {
    int count = 0;
    // 遍历所有合法移动并评分
    u64 temp_moves = moves;
    while (temp_moves) {
        int pos = __builtin_ctzll(temp_moves);
        temp_moves &= (temp_moves - 1);
        
        sorted_moves[count].pos = pos;
        sorted_moves[count].score = evaluate_move(board, pos);
        count++;
    }
    // 按分数排序
    qsort(sorted_moves, count, sizeof(ScoredMove), compare_moves);
    return count;
}

// uint64_t gen_moves(uint64_t my_pieces, uint64_t opp_pieces){
//     Board b = {{my_pieces, opp_pieces}};
//     return generate_moves(b);
// }

#define CORNER_MASK 0x8100000000000081ULL


static int midgame_search(Board board, int depth, int alpha, int beta);

static inline int middle_score(Board b) {
    // 权重
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

// Midgame negamax alpha-beta using middle_score for evaluation.
static int midgame_search(Board board, int depth, int alpha, int beta) {
    // Removed early cutoff on empties; only depth==0 triggers evaluation
    if (depth == 0) {
        return middle_score(board);
    }
    u64 moves = generate_moves(board);
    if (!moves) {
        // PASS：尝试对方是否也无棋
        Board opp = board; SWAP_BOARD(opp);
        u64 opp_moves = generate_moves(opp);
        if (!opp_moves) {
            // 双无可走，局面终结，用终局真实差值
            return final_score(board);
        }
        // 单方无棋：恢复为不减深策略
        return -midgame_search(opp, depth, -beta, -alpha);
    }
    // 着法排序（启发式提高剪枝）
    ScoredMove sorted_moves[32];
    int move_count = get_sorted_moves(board, moves, sorted_moves);
    int best = -INT_MAX;
    for (int i = 0; i < move_count; ++i) {
        int pos = sorted_moves[i].pos;
        Board nb = make_flips(board, pos);
        SWAP_BOARD(nb); // 轮到对手
        int val = -midgame_search(nb, depth - 1, -beta, -alpha);
        if (val > best) best = val;
        if (val > alpha) alpha = val;
        if (alpha >= beta) break; // 剪枝
    }
    return best;
}

int choose_move(uint64_t my_pieces, uint64_t opp_pieces) {
    int empties = 64 - COUNT_BITS(my_pieces | opp_pieces);
    if (empties <= ENDING_DEPTH) {
        int best_move = -1;
        // 调用终局枚举（局面以当前方视角传入）
        solve_endgame(my_pieces, opp_pieces, &best_move);
        return best_move; // 可能为 -1 (PASS)
    }
    Board board = (Board){{my_pieces, opp_pieces}};
    u64 moves = generate_moves(board);
    if (!moves) return -1; // PASS
    ScoredMove sorted_moves[32];
    int move_count = get_sorted_moves(board, moves, sorted_moves);
    if (move_count <= 0) return -1;
    int best_pos = sorted_moves[0].pos;
    int best_val = -INT_MAX;
    for (int i = 0; i < move_count; ++i) {
        int pos = sorted_moves[i].pos;
        Board nb = make_flips(board, pos);
        SWAP_BOARD(nb);
        int val = -midgame_search(nb, MIDGAME_DEPTH - 1, -INT_MAX, INT_MAX);
        if (val > best_val) { best_val = val; best_pos = pos; }
    }
    return best_pos;
}

// ---------------- Wrapper for project interface ----------------
// makeMove: required external interface. Converts 2D char board to bitboards, calls engine, returns chosen move.
// Signature MUST stay identical to legacy bots.
int makeMove(const char board[][26], int n, char turn, int *row, int *col) {
    if (n != 8) {
        // Simple fallback: scan for first legal move (8-direction Othello rule)
        int dr[8] = {-1,-1,-1,0,0,1,1,1};
        int dc[8] = {-1,0,1,-1,1,-1,0,1};
        char me = turn;
        char opp = (turn == 'B') ? 'W' : 'B';
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                if (board[r][c] != '.' && board[r][c] != 'U' && board[r][c] != 'u') continue;
                int legal = 0;
                for (int k = 0; k < 8 && !legal; ++k) {
                    int rr = r + dr[k], cc = c + dc[k];
                    int seen = 0;
                    while (rr >=0 && rr < n && cc >=0 && cc < n && board[rr][cc] == opp) { rr+=dr[k]; cc+=dc[k]; seen=1; }
                    if (seen && rr>=0 && rr<n && cc>=0 && cc<n && board[rr][cc]==me) legal = 1;
                }
                if (legal) { if (row) *row = r; if (col) *col = c; return 0; }
            }
        }
        return -1; // no move
    }
    // Bitboard conversion for 8x8
    uint64_t my = 0ULL, opp = 0ULL;
    if (turn == 'B') {
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                char ch = board[r][c];
                int idx = r*8 + c; // mapping: (r,c) -> bit idx
                if (ch == 'B' || ch == 'b') my |= (1ULL << idx);
                else if (ch == 'W' || ch == 'w') opp |= (1ULL << idx);
            }
        }
    } else { // turn == 'W'
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                char ch = board[r][c];
                int idx = r*8 + c;
                if (ch == 'W' || ch == 'w') my |= (1ULL << idx);
                else if (ch == 'B' || ch == 'b') opp |= (1ULL << idx);
            }
        }
    }
    int pos = choose_move(my, opp); // -1 means PASS
    if (pos < 0) return -1;
    if (row) *row = pos / 8;
    if (col) *col = pos % 8;
    return 0;
}
// ---------------- End wrapper ----------------