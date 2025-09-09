#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>

#define MIDGAME_DEPTH 5

typedef uint64_t u64;

typedef struct { u64 board[2]; } Board; // board[0]=当前方, board[1]=对方

typedef struct {
    int pos;
    int score;
} ScoredMove;

// --- begin: root sampling helpers ---
typedef struct { int pos; int eval; } EvalMove;

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
static inline double rng_double01(void) {
    return (splitmix64_next() >> 11) * (1.0 / 9007199254740992.0); // [0,1)
}

static int compare_eval_desc(const void* a, const void* b) {
    const EvalMove* x = (const EvalMove*)a;
    const EvalMove* y = (const EvalMove*)b;
    return (y->eval - x->eval); // 降序
}
// --- end: root sampling helpers ---

#define MY_PIECES(b)    ((b).board[0])
#define OPP_PIECES(b)   ((b).board[1])
#define ALL_PIECES(b)   (MY_PIECES(b) | OPP_PIECES(b))
#define EMPTY_SQUARES(b) (~ALL_PIECES(b))
#define COUNT_BITS(x)   __builtin_popcountll(x)
#define MY_COUNT(b)     COUNT_BITS(MY_PIECES(b))
#define OPP_COUNT(b)    COUNT_BITS(OPP_PIECES(b))
#define SWAP_BOARD(b) do { u64 t=(b).board[0]; (b).board[0]=(b).board[1]; (b).board[1]=t; } while(0)

// 初始棋型（标准 8x8 开局）：黑子在 (3,4),(4,3)，白子在 (3,3),(4,4)
#define INIT_BLACK ((1ULL<<28) | (1ULL<<35))
#define INIT_WHITE ((1ULL<<27) | (1ULL<<36))

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
    // 位置评分表（行优先，idx = r*8 + c）
    // 角=20；角三邻格=-10；其余边=8；内部=3；空=0（自然不加分）
    static const int PST[64] = {
        20, -10,  8,  8,  8,  8, -10, 20,
       -10, -10,  3,  3,  3,  3, -10,-10,
         8,   3,  3,  3,  3,  3,   3,  8,
         8,   3,  3,  3,  3,  3,   3,  8,
         8,   3,  3,  3,  3,  3,   3,  8,
         8,   3,  3,  3,  3,  3,   3,  8,
       -10, -10,  3,  3,  3,  3, -10,-10,
        20, -10,  8,  8,  8,  8, -10, 20
    };
    int myScore = 0, oppScore = 0;
    // 我方棋盘分（按位枚举）
    u64 my = MY_PIECES(b);
    while (my) {
        int idx = __builtin_ctzll(my);
        my &= (my - 1);
        myScore += PST[idx];
    }
    // 对方棋盘分
    u64 opp = OPP_PIECES(b);
    while (opp) {
        int idx = __builtin_ctzll(opp);
        opp &= (opp - 1);
        oppScore += PST[idx];
    }
    // 行动力：合法落子数 × 5
    int myMob = COUNT_BITS(generate_moves(b));
    Board ob = b; SWAP_BOARD(ob);
    int oppMob = COUNT_BITS(generate_moves(ob));
    return (myScore + 5 * myMob) - (oppScore + 5 * oppMob);
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

    /* Later this script is used for CHASE NN testing, so don't use endgame part */
    // if (empties <= 16) {
    //     int best_move = -1;
    //     solve_endgame(my_pieces, opp_pieces, &best_move);
    //     return best_move; // 可能为 -1 (PASS)
    // }

    Board board = (Board){{my_pieces, opp_pieces}};
    u64 moves = generate_moves(board);
    if (!moves) return -1; // PASS

    // 黑方第一步：初始布局时，4 个中心对称着法中等概率随机选择
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
        // 理论上不会走到这里，兜底继续后续逻辑
    }

    // 先对所有合法手做中局搜索评分
    ScoredMove sorted_moves[32];
    int move_count = get_sorted_moves(board, moves, sorted_moves);
    if (move_count <= 0) return -1;

    EvalMove evals[32];
    int best_idx = 0;
    int best_val = -INT_MAX;

    for (int i = 0; i < move_count; ++i) {
        int pos = sorted_moves[i].pos;
        Board nb = make_flips(board, pos);
        SWAP_BOARD(nb);
        int val = -midgame_search(nb, MIDGAME_DEPTH - 1, -INT_MAX, INT_MAX);
        evals[i].pos = pos;
        evals[i].eval = val;
        if (val > best_val) { best_val = val; best_idx = i; }
    }

    // 只有 1 手可走：直接返回
    if (move_count <= 1) {
        return evals[best_idx].pos;
    }

    // 按真实搜索得分降序排序
    qsort(evals, move_count, sizeof(EvalMove), compare_eval_desc);

    // 大差距保护：最优明显领先时不随机
    const int HARD_MARGIN = 16;
    if (move_count >= 2 && (evals[0].eval - evals[1].eval) >= HARD_MARGIN) {
        return evals[0].pos;
    }

    // 仅在 Top-3 内做“硬比例”采样（不再使用温度/softmax）
    int k = (move_count < 3) ? move_count : 3;

    // 开局 vs 中后期的固定比例（相对权重）
    double w[3] = {0};
    if (empties >= 44) {
        // 开局：8 : 1.5 : 0.5
        w[0] = 8.0; w[1] = (k >= 2 ? 1.5 : 0.0); w[2] = (k >= 3 ? 0.5 : 0.0);
    } else {
        // 中后期：9 : 0.7 : 0.3
        w[0] = 9.0; w[1] = (k >= 2 ? 0.7 : 0.0); w[2] = (k >= 3 ? 0.3 : 0.0);
    }

    double wsum = 0.0;
    for (int i = 0; i < k; ++i) wsum += w[i];
    if (wsum <= 0.0) return evals[0].pos; // 兜底

    // 采样
    double u = rng_double01() * wsum;
    double acc = 0.0;
    int pick = 0;
    for (int i = 0; i < k; ++i) {
        acc += w[i];
        if (u <= acc) { pick = i; break; }
    }
    return evals[pick].pos;
}
