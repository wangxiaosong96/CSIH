"""Matched random-edge or entity-disjoint evaluation with recorded outputs."""
import copy
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
try:
    from .ablation import ablation_config
    from .cold_start import make_cold_start_split
    from .generate import (load_dataset, build_training_graph, context_from_pairs,
                           sign_preserving_edge_dropout, corrupt_training_inputs, sha256_file)
    from .graph import CSIH
    from .parameter import get_args
    from .save_results import write_json, write_csv
    from .util import set_seed, evaluate_metrics, total_loss
except ImportError:
    from ablation import ablation_config
    from cold_start import make_cold_start_split
    from generate import (load_dataset, build_training_graph, context_from_pairs,
                          sign_preserving_edge_dropout, corrupt_training_inputs, sha256_file)
    from graph import CSIH
    from parameter import get_args
    from save_results import write_json, write_csv
    from util import set_seed, evaluate_metrics, total_loss


def validate_split(data, split, protocol):
    partitions = [np.asarray(split[key], dtype=int) for key in ("train_idx", "val_idx", "test_idx")]
    for indices in partitions:
        if indices.ndim != 1 or not len(indices) or len(set(indices)) != len(indices):
            raise ValueError("Every split partition requires distinct, nonempty row indices")
        if indices.min() < 0 or indices.max() >= len(data.pairs):
            raise ValueError("Split row index outside the input dataset")
        if len(np.unique(data.labels[indices].numpy())) != 2:
            raise ValueError("A split partition contains only one class; use a larger dataset/holdout or another predefined seed")
    if any(set(partitions[i]) & set(partitions[j]) for i in range(3) for j in range(i)):
        raise ValueError("Training, validation and test row indices overlap")
    if protocol == "random" and set(np.concatenate(partitions)) != set(range(len(data.pairs))):
        raise ValueError("A random-edge fold must partition the entire input dataset")
    if protocol != "random":
        tr, va, te = [data.pairs[idx].numpy() for idx in partitions]
        for col, name in ((0, "drug"), (1, "gene")):
            if protocol in (name, "double"):
                if set(te[:, col]) & (set(tr[:, col]) | set(va[:, col])):
                    raise ValueError(f"Held-out test {name} appears in training/model selection")
                if set(va[:, col]) & set(tr[:, col]):
                    raise ValueError(f"Validation {name} appears in training")


def make_splits(data, args, seed):
    result = []
    if args.protocol == "random":
        y = data.labels.numpy()
        if np.bincount(y).min() < args.folds:
            raise ValueError("Each sign class must have at least --folds observations")
        cv = StratifiedKFold(args.folds, shuffle=True, random_state=seed)
        for outer_train, test in cv.split(np.zeros(len(y)), y):
            train, val = train_test_split(outer_train, test_size=args.val_ratio,
                                          stratify=y[outer_train], random_state=seed)
            result.append({"train_idx": train, "val_idx": val, "test_idx": test})
    else:
        result = [make_cold_start_split(data.pairs.numpy(), args.protocol, args.holdout_ratio, seed, args.val_ratio)]
    loaded = []
    if args.splits:
        files = list((Path(args.splits) / args.protocol / f"seed_{seed}").glob("fold_*.json"))
        if len(files) != len(result):
            raise ValueError("Saved fold count differs from --folds/protocol; specify the original fold count")
    for fold, generated in enumerate(result, start=1):
        relative = Path(args.protocol) / f"seed_{seed}" / f"fold_{fold}.json"
        if args.splits:
            saved = json.loads((Path(args.splits) / relative).read_text(encoding="utf-8"))
            if saved.get("dataset_sha256") != data.fingerprint or saved.get("protocol") != args.protocol:
                raise ValueError("Saved split does not match this dataset byte hash/protocol")
            generated = saved
        validate_split(data, generated, args.protocol)
        payload = {**generated, "dataset_sha256": data.fingerprint, "protocol": args.protocol,
                   "seed": seed, "fold": fold, "row_indexing": "0-based input CSV data-row positions"}
        write_json(Path(args.output) / "splits" / relative, payload)
        loaded.append((fold, payload))
    if args.protocol == "random":
        test_rows = [int(i) for _, split in loaded for i in split["test_idx"]]
        if len(test_rows) != len(data.pairs) or len(set(test_rows)) != len(data.pairs):
            raise ValueError("Random-edge test folds must cover each observed pair exactly once per seed")
    return loaded


@torch.no_grad()
def evaluate(model, drug_features, gene_features, pos_edges, neg_edges, pairs, labels, context, batch_size):
    model.eval()
    encoded = model.encode(drug_features, gene_features, pos_edges, neg_edges, context_mask=context)
    probabilities = []
    for batch in pairs.split(batch_size):
        probabilities.append(model.decoder(encoded["drug"], encoded["gene"], batch).softmax(-1)[:, 1].cpu())
    probabilities = torch.cat(probabilities).numpy()
    return evaluate_metrics(labels.cpu().numpy(), probabilities), probabilities


def encoded_positive_view(model, df, gf, pos, neg, context, args):
    return model.encode(df, gf, sign_preserving_edge_dropout(pos, args.edge_drop),
                        sign_preserving_edge_dropout(neg, args.edge_drop), context_mask=context,
                        mask_rate=args.feature_mask, include_unseen=False)


def encoded_negative_views(model, df, gf, pos, neg, context, args):
    views = []
    for _ in range(args.num_negatives):
        cp, cn, permutations = corrupt_training_inputs(pos, neg, context, model.num_drugs,
                                                       args.negative_mode, args.sign_flip_rate)
        views.append(model.encode(df, gf, cp, cn, context_mask=context,
                                  permutations=permutations, include_unseen=False))
    return views


def train_fold(data, split, args, seed, fold, device, folder):
    set_seed((seed * 100003 + fold * 1009) % (2 ** 32))
    df, gf = data.drug_features.to(device), data.gene_features.to(device)
    train_idx, val_idx, test_idx = [np.asarray(split[k], dtype=int) for k in ("train_idx", "val_idx", "test_idx")]
    train_pairs, train_y = data.pairs[train_idx].to(device), data.labels[train_idx].to(device)
    pos, neg = build_training_graph(train_pairs, train_y)
    context = (torch.ones(len(df) + len(gf), dtype=torch.bool, device=device) if args.protocol == "random"
               else context_from_pairs(train_pairs, len(df), len(gf)))
    model = CSIH(args, len(df), len(gf), df.size(1), gf.size(1)).to(device)
    # Explicit L2 in total_loss: no second regularization term in the optimizer.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    best_auc, best_state, best_epoch, wait = -float("inf"), None, 0, 0
    history = []
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", vars(args))
    for epoch in range(args.epochs):
        # Keeps stochastic schedules matched across the full / w/o-loss controls.
        set_seed((seed * 100003 + fold * 1009 + epoch * 7919) % (2 ** 32))
        model.train()
        encoded = model.encode(df, gf, pos, neg, context_mask=context, include_unseen=False)
        logits = [model.decoder(encoded["drug"], encoded["gene"], batch) for batch in train_pairs.split(args.pair_batch_size)]
        out = {**encoded, "logits": torch.cat(logits)}
        if args.lambda_node or args.lambda_graph:
            view1 = encoded_positive_view(model, df, gf, pos, neg, context, args)
            view2 = encoded_positive_view(model, df, gf, pos, neg, context, args)
        else:
            view1 = view2 = None
        negative_views = encoded_negative_views(model, df, gf, pos, neg, context, args) if args.lambda_graph else []
        loss, parts = total_loss(out, train_y, view1, view2, args, negative_views, model, return_parts=True)
        graph_grad_norm = None
        if epoch == 0 and args.lambda_graph:
            encoder_parameters = [p for name, p in model.named_parameters() if not name.startswith("decoder.")]
            gradients = torch.autograd.grad(parts["graph"], encoder_parameters, retain_graph=True, allow_unused=True)
            graph_grad_norm = float(torch.sqrt(sum((g.square().sum() for g in gradients if g is not None),
                                                   loss.new_tensor(0))).detach().cpu())
            if not np.isfinite(graph_grad_norm) or graph_grad_norm <= 1e-12:
                raise RuntimeError("Graph objective has no useful encoder gradient. Check corruption mode/data; do not report this run.")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        validation, _ = evaluate(model, df, gf, pos, neg, data.pairs[val_idx].to(device), data.labels[val_idx],
                                  context, args.pair_batch_size)
        entry = {"epoch": epoch + 1, **{key: float(value.detach().cpu()) for key, value in parts.items()},
                 "validation_AUC": validation["AUC"], "graph_encoder_gradient_norm": graph_grad_norm}
        history.append(entry)
        if validation["AUC"] > best_auc:
            best_auc, best_state, best_epoch, wait = validation["AUC"], copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            wait += 1
        if wait >= args.patience:
            break
    write_json(folder / "training_history.json", history)
    model.load_state_dict(best_state)
    metrics, probabilities = evaluate(model, df, gf, pos, neg, data.pairs[test_idx].to(device), data.labels[test_idx],
                                      context, args.pair_batch_size)
    torch.save({"state_dict": best_state, "config": vars(args), "dataset_sha256": data.fingerprint,
                "drug_names": data.drug_names, "gene_names": data.gene_names,
                "split": json.loads(json.dumps(split, default=lambda value: value.tolist())),
                "context_mask": context.cpu(), "selected_epoch": best_epoch}, folder / "checkpoint.pt")
    records = []
    for row_index, probability in zip(test_idx, probabilities):
        di, gi = data.pairs[row_index].tolist()
        records.append({"dataset": Path(args.data).stem, "protocol": args.protocol, "variant": args.variant,
                        "seed": seed, "fold": fold, "row_index": int(row_index),
                        "drug": data.drug_names[di], "gene": data.gene_names[gi],
                        "true_sign": 1 if data.labels[row_index] else -1,
                        "prob_positive": float(probability), "prob_negative": float(1 - probability),
                        "predicted_sign": 1 if probability >= 0.5 else -1})
    write_csv(folder / "test_predictions.csv", records)
    write_json(folder / "test_metrics.json", metrics)
    return metrics


def main(argv=None):
    args = get_args(argv)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty --output directory; existing experimental results are never overwritten")
    configurations = [ablation_config(args, variant) for variant in args.variants]
    need_d = any(cfg.use_drug_features for cfg in configurations)
    need_g = any(cfg.use_gene_features for cfg in configurations)
    if need_d and not args.drug_features or need_g and not args.gene_features:
        raise ValueError("Retained foundation-model inputs require real precomputed feature files; no identity-matrix fallback")
    data = load_dataset(args.data, args.drug_features if need_d else None, args.gene_features if need_g else None,
                        args.drug_ids, args.gene_ids)
    torch.set_num_threads(args.threads)
    chosen = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(chosen)
    manifest = {"implementation": "CSIH corrected 2026-11-27", "dataset_sha256": data.fingerprint,
                "data": str(args.data), "arguments": vars(args), "python": platform.python_version(),
                "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                "device": str(device), "drug_names": data.drug_names, "gene_names": data.gene_names,
                "positive_class": "+1", "AUPR_definition": "sklearn average_precision_score",
                "Binary_F1_threshold": 0.5, "validation_selection": "maximum validation AUC",
                "feature_sha256": {key: sha256_file(path) for key, path in
                    (("drug", args.drug_features if need_d else None), ("gene", args.gene_features if need_g else None)) if path}}
    manifest["source_file_sha256"] = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    write_json(output / "manifest.json", manifest)
    write_json(output / "entity_mappings.json", {"drugs": data.drug_names, "genes": data.gene_names})
    all_splits = {seed: make_splits(data, args, seed) for seed in args.seeds}
    fold_rows, seed_rows, summaries = [], [], []
    for cfg in configurations:
        write_json(output / cfg.variant / "config.json", vars(cfg))
        for seed in args.seeds:
            current = []
            for fold, split in all_splits[seed]:
                folder = output / cfg.variant / f"seed_{seed}" / f"fold_{fold}"
                metrics = train_fold(data, split, cfg, seed, fold, device, folder)
                row = {"dataset": Path(args.data).stem, "protocol": args.protocol, "method": cfg.variant,
                       "seed": seed, "fold": fold, **metrics}
                fold_rows.append(row)
                current.append(metrics)
                print(json.dumps(row), flush=True)
                write_csv(output / "fold_metrics.csv", fold_rows)
            seed_rows.append({"dataset": Path(args.data).stem, "protocol": args.protocol, "method": cfg.variant,
                              "seed": seed, **{key: float(np.mean([m[key] for m in current])) for key in current[0]}})
            write_csv(output / "seed_scores.csv", seed_rows)
        rows = [r for r in seed_rows if r["method"] == cfg.variant]
        for metric in ("AUC", "AUPR", "Binary-F1"):
            values = [r[metric] for r in rows]
            summaries.append({"dataset": Path(args.data).stem, "protocol": args.protocol, "method": cfg.variant,
                              "metric": metric, "mean": float(np.mean(values)),
                              "SD": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                              "n_seeds": len(values)})
    write_json(output / "summary.json", summaries)


if __name__ == "__main__":
    main()
