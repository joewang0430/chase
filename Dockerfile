# Bot build environment (Ubuntu 16.04 base) to produce Linux x86_64 ELF .so for Botzone
# Replaces earlier generic python:3.6-slim image.

# Use X86_64 base image explicitly
FROM --platform=linux/amd64 ubuntu:16.04

# 1. Install toolchain + Python (minimal) for build & quick test
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential python3 ca-certificates file \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy sources (expects botzone/data/ending.c and botzone/test_main.py etc.)
COPY . .

# Ensure unified /app/data path (symlink to botzone/data for legacy relative lookups)
RUN mkdir -p botzone/data && ln -s botzone/data data

# 3. Compile shared library at correct path - with compatibility fixes
RUN sed 's/#pragma GCC target.*//g' botzone/data/ending.c > botzone/data/ending_compat.c && \
    gcc -O3 -std=c11 -fPIC -shared botzone/data/ending_compat.c -o botzone/data/ending.so \
    && file botzone/data/ending.so \
    && nm -D botzone/data/ending.so | grep -E "solve_endgame$" || true

# 4. Debug-enhanced ctypes load test (absolute + relative)
RUN echo '--- ls /app ---' && ls -l /app && \
    echo '--- ls /app/botzone/data ---' && ls -l /app/botzone/data && \
    echo '--- ls /app/data (symlink) ---' && ls -l /app/data && \
    python3 - <<'PYTEST'
import ctypes, os, sys
candidates = [
    'data/ending.so',
    'botzone/data/ending.so',
]
print('[CTYPES-TEST] cwd=', os.getcwd())
for p in candidates:
    print('  candidate', p, 'exists=', os.path.exists(p))
loaded = False
for p in candidates:
    if os.path.exists(p):
        try:
            lib = ctypes.CDLL(os.path.abspath(p))
            print('[CTYPES-TEST] loaded from', p, 'has solve_endgame=', hasattr(lib,'solve_endgame'))
            loaded = True
            break
        except OSError as e:
            print('[CTYPES-TEST] failed loading', p, e, file=sys.stderr)
if not loaded:
    print('[CTYPES-TEST] ERROR: no candidate could be loaded', file=sys.stderr)
    sys.exit(1)
PYTEST

CMD ["bash"]

# ============================= MAC 使用流程 (完整命令链) =============================
# 目录假设：当前就在 chase 根目录 (含本 Dockerfile)。
# 产生的 .so 目标：botzone/data/ending.so  (Linux x86_64 ELF)
# -----------------------------------------------------------------------------------
# 1. 构建镜像
#   Intel Mac:
#       docker build -t chase-bot .
#   Apple Silicon (M1/M2 等，必须强制 x86_64):
#       docker build --platform=linux/amd64 -t chase-bot .
#
# 2. 将容器内编译好的 .so 拷贝到宿主 (保持原路径层级)
#       docker run --rm -v "$PWD":/out chase-bot cp /app/botzone/data/ending.so /out/botzone/data/
#   若宿主不存在 botzone/data 目录，先:  mkdir -p botzone/data
#
# 3. 验证文件格式 (在宿主 Mac 上执行，确认是 Linux x86-64 ELF)
#       file botzone/data/ending.so
#     期望包含: "ELF 64-bit LSB shared object, x86-64"
#
# 4. 确认导出符号 solve_endgame 存在 (nm 输出含 T solve_endgame)
#       docker run --rm chase-bot nm -D /app/botzone/data/ending.so | grep solve_endgame
#     (或在宿主安装了 binutils-for-elf 也可直接 nm -D，但容器内更稳)
#
# 5. 容器内再做一次 ctypes 载入验证 (确保可动态加载)
#       docker run --rm chase-bot python3 - <<'PY'
#       import ctypes; lib=ctypes.CDLL('botzone/data/ending.so');
#       print('loaded, has solve_endgame =', hasattr(lib,'solve_endgame'))
#       PY
#
# 6. 上传 botzone/data/ending.so 到 Botzone 平台的 data/ 目录。
#    Python 侧加载路径若使用 data/ending.so 也可工作（因本地已建立 data -> botzone/data 链接，可在平台同步该结构或直接改成 botzone/data/ 路径）。
#
# 7. (可选) 清理本地悬空镜像层:
#       docker image prune -f
# -----------------------------------------------------------------------------------
# 若需重新编译：修改 C 源后重复步骤 1~5。
# ==============================================================================

# Extract built .so to host (after build):
#   docker run --rm -v "$PWD":/out chase-bot cp /app/botzone/data/ending.so /out/botzone/data/
# Verify format on host:
#   file botzone/data/ending.so   # should show: ELF 64-bit LSB shared object, x86-64

