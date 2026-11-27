"""Executable defaults, recorded per run; tune on validation data only."""
import argparse
try:
    from .ablation import VARIANTS
except ImportError:
    from ablation import VARIANTS


def get_args(argv=None):
    p = argparse.ArgumentParser("CSIH corrected implementation")
    p.add_argument("--data", required=True)
    p.add_argument("--drug_features")
    p.add_argument("--gene_features")
    p.add_argument("--drug_ids")
    p.add_argument("--gene_ids")
    p.add_argument("--output", default="results/new_run")
    p.add_argument("--splits", help="Directory containing matching saved splits/<protocol>/seed_*/fold_*.json")
    p.add_argument("--protocol", choices=("random", "drug", "gene", "double"), default="random")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=["full"])
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--identity_dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--n_intents", type=int, default=4)
    p.add_argument("--intent_dim", type=int, default=32)
    p.add_argument("--n_prototypes", type=int, default=128)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--hypergraph_temperature", type=float, default=0.2)
    p.add_argument("--graph_temperature", type=float, default=0.2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--edge_drop", type=float, default=0.1)
    p.add_argument("--feature_mask", type=float, default=0.1)
    p.add_argument("--negative_mode", choices=("both", "feature_shuffle", "sign_flip"), default="both")
    p.add_argument("--sign_flip_rate", type=float, default=0.2)
    p.add_argument("--num_negatives", type=int, default=1)
    p.add_argument("--lambda_node", type=float, default=1e-5, help="lambda_1")
    p.add_argument("--lambda_graph", type=float, default=1e-5, help="lambda_2")
    p.add_argument("--lambda_decor", type=float, default=1e-4, help="lambda_3")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="Explicit beta * ||Theta||^2; Adam weight_decay=0")
    p.add_argument("--batch_size", type=int, default=512, help="Maximum sampled nodes for node InfoNCE")
    p.add_argument("--pair_batch_size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--holdout_ratio", type=float, default=0.2)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    a = p.parse_args(argv)
    positive = ("threads", "hidden_dim", "identity_dim", "n_intents", "intent_dim", "n_prototypes",
                "topk", "batch_size", "pair_batch_size", "epochs", "patience", "num_negatives")
    if any(getattr(a, k) <= 0 for k in positive) or a.layers < 0 or a.folds < 2 or a.batch_size < 2:
        p.error("Invalid count/dimension; folds and node batch size must be >=2")
    if min(a.temperature, a.graph_temperature, a.hypergraph_temperature, a.lr, a.grad_clip) <= 0:
        p.error("Temperatures, learning rate and grad_clip must be positive")
    if min(a.lambda_node, a.lambda_graph, a.lambda_decor, a.weight_decay) < 0:
        p.error("Loss coefficients must be nonnegative")
    if any(not 0 <= v < 1 for v in (a.dropout, a.edge_drop, a.feature_mask)):
        p.error("Drop rates must be in [0,1)")
    if not 0 < a.val_ratio < 1 or not 0 < a.holdout_ratio < 1 or not 0 < a.sign_flip_rate <= 1:
        p.error("Invalid split/corruption ratios")
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.variants)) != len(a.variants) or min(a.seeds) < 0:
        p.error("Seeds/variants must be unique and seeds nonnegative")
    return a
