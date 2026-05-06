# Paged Attention Performance Workload Design

## Goal

Extend the static-template performance benchmark so it can measure the full
paged attention graph, not only the small BGEMM manual-scope template case.

## Design

The benchmark keeps the existing `static_hit` versus `traditional_baseline`
comparison and adds a workload selector. `bgemm_manual_scope` remains available
as a small smoke workload. `paged_attention` uses
`examples.models.paged_attention.build_paged_attention_program`, which exercises
the full pipeline:

```text
kernel_init_inplace -> kernel_qk_matmul -> kernel_softmax_prepare
  -> kernel_pv_matmul -> kernel_online_update
```

Paged attention tensor construction reuses
`examples.models.paged_attention.build_tensors` so the benchmark follows the
same graph and tensor layout as the maintained example. CLI parameters expose
the key shape knobs: batch, number of heads, head dimension, block size, context
length, max model length, and scale.

## Output

`summary.json` records the selected workload and workload config. `summary.md`
prints the workload and parsed runtime rounds next to the existing compile and
device timing metrics, making it easier to spot log parsing mismatches.

## Testing

This change is intended for the hardware system-test environment. Local pytest
is not required for this environment-limited benchmark update; use static diff
checks locally and run the benchmark command on the target server.
