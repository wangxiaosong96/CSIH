# Implementation changes

Base reviewed: `wangxiaosong96/CSIH`, commit `cf144af63ad07300a43c9fee844fa231bc3091be`.

This is a corrected, executable source revision. It is not a recovered copy of an unavailable private training implementation and does not certify previously reported benchmark values.

| Files | Change |
|---|---|
| `src/util.py`, `src/run.py`, `src/generate.py` | Replace degenerate embedding-row permutation negatives with corrupted inputs/edges re-encoded by the shared network; check actual encoder gradients |
| `src/preprocess.py`, `configs/dgidb_sign_mapping.json` | Exact token matching, conflict/deduplication policy and complete per-export audit |
| `src/prepare_drugbank.py` | Explicit released expression-label import; remove pharmacological action parsing |
| `src/prepare_lincs.py` | Separate released-label import and auditable condition-score reconstruction |
| `src/ablation.py`, `src/graph.py` | Real EF, FM, ID, SIH, MI and objective ablations |
| `src/algorithm.py`, `src/graph.py` | Symmetric edge normalization; sparse incidence; intent-space residual; initial-aware gating; four pair terms; two logits |
| `src/cold_start.py`, `src/graph.py`, `src/run.py` | Disjoint held-out validation/test entities; training-context-only aggregation; independent query inference |
| `src/generate.py`, `src/feature_io.py`, extractors | ID-aligned feature loading; no silent ID-matrix FM fallback; real optional extraction adapters |
| `src/baseline.py`, `src/paired_statistics.py` | Training-only frequency methods and seed-paired uncertainty/statistics |
| `src/plot_results.py` | Calculate curve legend metrics from actual predictions |
| `src/save_results.py`, `src/run.py` | Preserve configurations, source/data hashes, splits, gradients, checkpoints and row predictions |
| `tests/`, `src/check_graph_loss.py`, `src/smoke_test.py` | Functional regression tests and explicitly synthetic integration exercises |

Behavioral conventions made concrete in this implementation:

1. Early fusion is arithmetic-mean fusion, retaining both hypergraph branches and their parameter count.
2. Graph InfoNCE has two symmetric positive anchors and one or more independently corrupted graph views; default raw corruption combines within-type input shuffling with 20% training-edge sign flips.
3. Channel matrices are column-normalized for decorrelation; both off-diagonal channel orders are included and sign-branch losses are averaged.
4. L2 is an explicit objective term, with no duplicate optimizer weight decay.
5. Held-out query representations read training-context hyperedge messages and degrees, without updating them.
6. Metrics use positive class +1, average precision for AUPR and a fixed 0.5 probability threshold for Binary-F1.
7. Saved old checkpoints are incompatible because the decoder, hypergraph residual and fusion definitions were corrected.

Align the revised Methods and coefficients to these conventions, or adjust them using the actual verified experimental design before rerunning. Full benchmark comparisons, optional large pretrained-model extraction, and GPU resource measurements were not rerun for this source delivery. Data, mappings and checkpoints must come from real experiment records.
