import os, time, torch
from nn.model_def import Net

def main():
    # Limit to single CPU thread (PyTorch + BLAS libs)
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    device = torch.device('cpu')
    model = Net().to(device).eval()

    BATCH = 1
    N = 1000
    x = torch.randn(BATCH, 3, 8, 8, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            model(x)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            model(x)
    end = time.perf_counter()

    total = end - start
    avg = total / N

    print(f"Device: {device}")
    print(f"Threads: {torch.get_num_threads()}")
    print(f"Runs: {N}  Batch: {BATCH}")
    print(f"Total time: {total:.6f} s")
    print(f"Avg per forward: {avg*1000:.3f} ms")
    print(f"Throughput: {N/total:.2f} it/s")

if __name__ == '__main__':
    main()
