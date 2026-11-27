# Validation record: 2026-11-27

These are functional checks on synthetic inputs. They do not reproduce the DGIdb, DrugBank or LINCS benchmark experiments and must not be used as paper performance evidence.

## Passed checks

1. Python compilation of the source modules and tests.
2. `python -m unittest discover -s tests -v`: **25 tests passed**, including all 11 variant subcases.
3. Standalone graph-loss gradient diagnostic: all checked encoder groups receive nonzero gradients.
4. Synthetic end-to-end run: **44 completed random-edge fits** (11 configurations, two seeds, two folds; two epochs per fit).
5. Synthetic unseen-drug, unseen-gene and double-cold-start evaluation: **3 completed fits**.
6. All four sign-frequency baselines using those recorded random-edge splits.
7. Paired seed-score statistics with CI, effect sizes and joint Holm correction.
8. ROC/PR plotting directly from saved synthetic test predictions.
9. After final split/metadata changes: **two additional random-edge folds** reusing the saved exact splits; checkpoints verified readable with `torch.load(..., weights_only=True)`.

Unit tests specifically check antagonist/inverse-agonist polarity, mixed-sign annotations, conflicting pairs, duplicate leakage, ID alignment, missing feature-row metadata, actual SIH removal, absence of ID parameters, invariance to removed FM inputs, symmetric graph normalization, training-only negative corruption, held-out entity partitioning, zero held-out feature/ID training gradients, independence among unseen queries, condition conflict reconstruction, training-only prior fallbacks and matched-seed statistical requirements.

## Standalone graph-loss diagnostic

Fixed synthetic seed: 37. PyTorch: 2.5.1+cpu.

| Quantity | Observed value |
|---|---:|
| Graph contrastive loss | 0.70029175 |
| Feature encoder gradient norm | 0.14294620 |
| Relation encoder gradient norm | 0.00996342 |
| Positive hypergraph gradient norm | 0.01241011 |
| Negative hypergraph gradient norm | 0.01703798 |

These norms come from the graph objective alone. The previous row-permutation/mean-pooling loss had no useful gradient; the corrected corruption is applied before re-encoding. Each real run repeats a first-epoch gradient check on its own data rather than relying only on this synthetic example.

## Scope

Tests used Python 3.12 on CPU. No real benchmark data or original pretrained feature files were supplied for this code correction. No full benchmark training, comparison with official learned baselines, GPU timing/memory measurement, or download/extraction with Uni-Mol/ESM-2 weights was performed. Optional extractor interfaces were implemented from the official APIs, but their external checkpoints and conformer/sequence inputs must be verified in the actual experiment environment.

Use `python -m src.smoke_test --output <new-directory>` to regenerate the synthetic integration record locally. Its output filenames and metadata explicitly mark the data as synthetic.
