"""Synthetic gradient diagnostic, not a benchmark performance experiment."""
import argparse
import json
import torch
try:
    from .parameter import get_args
    from .ablation import ablation_config
    from .generate import build_training_graph
    from .graph import CSIH
    from .run import encoded_positive_view, encoded_negative_views
    from .util import graph_contrastive_loss, set_seed
    from .save_results import write_json
except ImportError:
    from parameter import get_args
    from ablation import ablation_config
    from generate import build_training_graph
    from graph import CSIH
    from run import encoded_positive_view, encoded_negative_views
    from util import graph_contrastive_loss, set_seed
    from save_results import write_json


def run_diagnostic():
    set_seed(37)
    torch.set_num_threads(1)
    args = ablation_config(get_args(["--data", "SYNTHETIC_DIAGNOSTIC", "--hidden_dim", "16", "--intent_dim", "6",
                                    "--n_intents", "2", "--n_prototypes", "7", "--topk", "3", "--dropout", "0"]), "full")
    pairs = torch.tensor([[d, g] for d in range(8) for g in range(9) if (d + 2 * g) % 4 != 0])
    labels = torch.tensor([(d * 3 + g) % 2 for d, g in pairs.tolist()])
    pos, neg = build_training_graph(pairs, labels)
    df, gf = torch.randn(8, 10), torch.randn(9, 12)
    model = CSIH(args, 8, 9, 10, 12)
    model.train()
    context = torch.ones(17, dtype=torch.bool)
    first = encoded_positive_view(model, df, gf, pos, neg, context, args)
    second = encoded_positive_view(model, df, gf, pos, neg, context, args)
    negative = encoded_negative_views(model, df, gf, pos, neg, context, args)
    loss = graph_contrastive_loss(first["all"], second["all"], [v["all"] for v in negative], args.graph_temperature)
    model.zero_grad(set_to_none=True)
    loss.backward()
    norms = {}
    for name in ("feature_encoder", "relation_encoder", "pos_hypergraph", "neg_hypergraph"):
        squares = [p.grad.square().sum() for p in getattr(model, name).parameters() if p.grad is not None]
        norms[name] = float(torch.sqrt(torch.stack(squares).sum())) if squares else 0.0
    if not torch.isfinite(loss) or not all(value > 1e-10 for value in norms.values()):
        raise AssertionError(f"Graph loss failed gradient diagnostic: {norms}")
    return {"test_type": "synthetic functional gradient check; not paper benchmark results",
            "seed": 37, "torch_version": torch.__version__, "graph_loss": float(loss.detach()),
            "encoder_gradient_norms": norms, "passed": True}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_diagnostic()
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
