"""GPU-side profile: where does a reservoir decision actually spend CUDA time?

Breaks one values() call into SpMM / GEMM / tanh / concat / readout /
host-overhead using torch.profiler (kineto/CUPTI). Run at slice + full size,
batch 1 and 32. Evidence for the fuse-or-not decision.
Usage: profile_gpu.py [--n 10000|full] [--batch 1] [--reps 30]
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np


def parse(a):
    o = dict(n="10000", batch=1, reps=30)
    i = 0
    while i < len(a):
        if a[i].startswith("--") and i + 1 < len(a):
            o[a[i][2:]] = a[i + 1]
            i += 2
        else:
            i += 1
    o["batch"] = int(o["batch"])
    o["reps"] = int(o["reps"])
    return o


def main():
    o = parse(sys.argv[1:])
    import scipy.sparse as sp
    import torch
    from torch.profiler import ProfilerActivity, profile
    assert torch.cuda.is_available(), "no CUDA"
    from brain import flybrain as F
    W = F.load_connectome().tocsr()
    if o["n"] != "full":
        W = W[:int(o["n"]), :int(o["n"])].tocsr()
    n = W.shape[0]
    dim = 52
    rng = np.random.default_rng(0)
    Wsp = torch.sparse_csr_tensor(
        torch.from_numpy(W.indptr).cuda(), torch.from_numpy(W.indices).cuda(),
        torch.from_numpy(W.data).float().cuda(), size=W.shape).cuda()
    Win = torch.from_numpy(rng.standard_normal((n, dim)).astype(np.float32) * 0.5).cuda()
    w = torch.from_numpy(rng.standard_normal(n + dim + 1).astype(np.float32) * 0.01).cuda()
    B = o["batch"]
    feats = torch.from_numpy(rng.standard_normal((dim, B)).astype(np.float32)).cuda()

    def step():
        u = feats
        x = torch.zeros((n, B), dtype=torch.float32, device="cuda")
        for _ in range(2):
            x = torch.tanh(torch.matmul(Wsp, x) + torch.matmul(Win, u))
        phi = torch.cat([x, u, torch.ones((1, B), dtype=torch.float32, device="cuda")], dim=0)
        return torch.matmul(w, phi)

    step()  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(o["reps"]):
        step()
    torch.cuda.synchronize()
    print(f"wall: n={n} batch={B}: {(time.time() - t0) / o['reps'] * 1000:.2f} ms/call", flush=True)

    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=acts, record_shapes=False) as prof:
        for _ in range(max(5, o["reps"] // 3)):
            step()
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="cuda_time_total", row_limit=14), flush=True)
    print("PROFILE-DONE", flush=True)


if __name__ == "__main__":
    main()
