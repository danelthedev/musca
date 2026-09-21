"""GPU-batched 1-ply decisions. Training scaffold (P0): same math as
value.Head, candidates encoded on CPU, reservoir + readout batched on GPU
(float32). Rule: GPU picks must equal CPU picks (parity gate) before any
training runs on this path.
"""
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class GPUHead:
    def __init__(self, W_scipy, Win, w, steps=2, device="cuda"):
        assert _HAS_TORCH, "torch required"
        self.n = W_scipy.shape[0]
        self.dim = Win.shape[1]
        self.steps = steps
        self.dev = device
        W = W_scipy.tocsr()
        self.Wsp = torch.sparse_csr_tensor(
            torch.from_numpy(W.indptr).to(device),
            torch.from_numpy(W.indices).to(device),
            torch.from_numpy(W.data).float().to(device),
            size=W.shape, device=device)
        self.Win_t = torch.from_numpy(np.ascontiguousarray(Win)).float().to(device)
        self.w_t = torch.from_numpy(np.ascontiguousarray(w)).float().to(device)

    def values(self, feats):
        """feats: (B, dim) float64 numpy -> (B,) float64 numpy values."""
        B = feats.shape[0]
        with torch.no_grad():
            u = torch.from_numpy(np.ascontiguousarray(feats)).float().to(self.dev).t()
            x = torch.zeros((self.n, B), dtype=torch.float32, device=self.dev)
            for _ in range(self.steps):
                x = torch.tanh(torch.matmul(self.Wsp, x) + torch.matmul(self.Win_t, u))
            phi = torch.cat([x, u, torch.ones((1, B), dtype=torch.float32, device=self.dev)], dim=0)
            v = torch.matmul(self.w_t, phi)
        return v.float().cpu().numpy().astype(float)

    def phi_batch(self, feats):
        """feats: (B, dim) numpy -> (B, n+dim+1) numpy full phi rows. Batch
        featurization for TD traces: one GPU pass per game."""
        B = feats.shape[0]
        with torch.no_grad():
            u = torch.from_numpy(np.ascontiguousarray(feats)).float().to(self.dev).t()
            x = torch.zeros((self.n, B), dtype=torch.float32, device=self.dev)
            for _ in range(self.steps):
                x = torch.tanh(torch.matmul(self.Wsp, x) + torch.matmul(self.Win_t, u))
            phi = torch.cat([x, u, torch.ones((1, B), dtype=torch.float32, device=self.dev)], dim=0)
        return phi.t().cpu().numpy().astype(float)

    def choose_batch(self, items, rng):
        """items: list of (g, moves). Returns list of picked moves.
        Same rule as Head.choose: white max, black min, 1e-9 ties via rng."""
        from . import features as FT
        feats, owners = [], []
        for ei, (g, moves) in enumerate(items):
            for m in moves:
                c = g.clone()
                c.apply(m)
                feats.append(FT.encode_state(c.board, c.bar, c.off, c.turn))
                owners.append((ei, g.turn, m))
        vals = self.values(np.array(feats))
        best, bestv = {}, {}
        for (ei, turn, m), v in zip(owners, vals):
            if ei not in best:
                best[ei], bestv[ei] = [m], v
            elif (turn == 0 and v > bestv[ei] + 1e-9) or (turn == 1 and v < bestv[ei] - 1e-9):
                best[ei], bestv[ei] = [m], v
            elif abs(v - bestv[ei]) <= 1e-9:
                best[ei].append(m)
        return [rng.choice(best[ei]) for ei in range(len(items))]
