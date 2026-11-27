"""Small synthetic end-to-end exercise; not a scientific benchmark rerun."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
try:
    from .ablation import VARIANTS
    from .save_results import write_json
except ImportError:
    from ablation import VARIANTS
    from save_results import write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True, help="New directory for explicitly SYNTHETIC data/results")
    a = parser.parse_args()
    output = Path(a.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty smoke-test directory")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    data = pd.DataFrame([{"drug": f"SYNTHETIC_D{d:02}", "gene": f"SYNTHETIC_G{g:02}",
                          "sign": 1 if (d + g) % 2 else -1}
                         for d in range(20) for g in range(24) if (d + 2 * g) % 5 != 0])
    data.to_csv(output / "SYNTHETIC_ONLY.csv", index=False)
    for kind, dim in (("drug", 5), ("gene", 7)):
        ids = sorted(data[kind].unique())
        np.savez(output / (kind + ".npz"), ids=np.asarray(ids), features=rng.normal(size=(len(ids), dim)).astype(np.float32))
    root = Path(__file__).resolve().parents[1]
    commands = []

    def run(module, arguments):
        cmd = [sys.executable, "-m", "src." + module, *map(str, arguments)]
        commands.append(cmd)
        with (output / f"step_{len(commands):02d}_{module}.log").open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)

    common = ["--data", output / "SYNTHETIC_ONLY.csv", "--drug_features", output / "drug.npz",
              "--gene_features", output / "gene.npz", "--device", "cpu", "--threads", "1",
              "--hidden_dim", "12", "--identity_dim", "4", "--intent_dim", "4", "--n_intents", "2",
              "--n_prototypes", "5", "--topk", "2", "--layers", "1", "--epochs", "2", "--patience", "2",
              "--batch_size", "32", "--pair_batch_size", "64", "--val_ratio", ".25", "--dropout", "0"]
    run("run", common + ["--output", output / "random", "--folds", "2", "--seeds", "7", "11",
                          "--variants", *VARIANTS])
    run("baseline", ["--data", output / "SYNTHETIC_ONLY.csv", "--splits", output / "random/splits",
                     "--output", output / "priors"])
    run("paired_statistics", ["--inputs", output / "random/seed_scores.csv", output / "priors/seed_scores.csv",
                              "--reference", "full", "--comparators", "CSIH-EF", "Drug-Gene-Prior",
                              "--output", output / "paired_synthetic.csv"])
    for protocol in ("drug", "gene", "double"):
        run("run", common + ["--output", output / protocol, "--protocol", protocol, "--seeds", "7"])
    files = sorted((output / "random/full").glob("seed_7/fold_*/test_predictions.csv"))
    run("plot_results", ["--predictions", *files, "--seed", "7", "--output", output / "SYNTHETIC_ROC_PR.png"])
    fold_table = pd.read_csv(output / "random/fold_metrics.csv")
    assert len(fold_table) == len(VARIANTS) * 2 * 2
    assert set(fold_table.method) == set(VARIANTS)
    for history in (output / "random").glob("*/seed_*/fold_*/training_history.json"):
        rows = json.loads(history.read_text())
        if "w/o_GCL" not in history.parts:
            assert rows[0]["graph_encoder_gradient_norm"] > 1e-12
    report = {"test_type": "SYNTHETIC functional/integration checks only", "passed": True,
              "random_edge_variant_count": len(VARIANTS), "random_edge_seed_count": 2,
              "random_edge_fold_count": 2, "random_edge_completed_fits": len(fold_table),
              "cold_start_completed_fits": 3, "frequency_baselines": 4,
              "paired_statistics_and_raw_prediction_plots": "passed", "commands": commands}
    write_json(output / "SMOKE_TEST_REPORT.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "commands"}, indent=2))


if __name__ == "__main__":
    main()
