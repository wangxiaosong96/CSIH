import argparse


def get_args():
    parser = argparse.ArgumentParser("CSIH")

    parser.add_argument("--data", required=True)
    parser.add_argument("--drug_features", default=None)
    parser.add_argument("--gene_features", default=None)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--identity_dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--n_intents", type=int, default=4)
    parser.add_argument("--intent_dim", type=int, default=32)
    parser.add_argument("--n_prototypes", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--graph_temperature", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--edge_drop", type=float, default=0.1)

    parser.add_argument("--lambda_sup", type=float, default=1.0)
    parser.add_argument("--lambda_node", type=float, default=1e-5)
    parser.add_argument("--lambda_graph", type=float, default=1e-5)
    parser.add_argument("--lambda_decor", type=float, default=1e-4)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5]
    )
    return parser.parse_args()
