# slice_10k: central-brain intrinsic core

- pool: superclass == cb_intrinsic (32164 units)
- rule: top 10000 by in-slice in+out degree (recurrent core)
- slice: n=10000 nnz=2248736 density=0.0225
- in-slice degree: min=451 med=595 max=8340
- dropped pool units: 22164 (low-degree pass-through)
- indices: slice_10k_idx.npy (matrix rows == neurons.parquet rows, verified)
- honesty: parquet has no mushroom-body/central-complex labels; this is
  central-brain association tissue, not a mapped MB circuit.
- (0.4s)
