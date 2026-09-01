import csv
from pathlib import Path


def append_result(path, dataset, seed, fold, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "seed", "fold", "AUC", "AUPR", "Binary-F1"]
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            "dataset": dataset,
            "seed": seed,
            "fold": fold,
            "AUC": metrics["AUC"],
            "AUPR": metrics["AUPR"],
            "Binary-F1": metrics["Binary-F1"],
        })
