"""Regression checks for scientific correctness, not benchmark-score fixtures."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.ablation import ablation_config, VARIANTS
from src.algorithm import BipartiteSignAggregator
from src.baseline import SignFrequencyBaseline
from src.check_graph_loss import run_diagnostic
from src.cold_start import make_cold_start_split
from src.generate import load_dataset, read_features, build_training_graph, corrupt_training_inputs
from src.graph import CSIH
from src.paired_statistics import holm_adjust, compare_seed_scores
from src.parameter import get_args
from src.prepare_lincs import audit_conditions
from src.preprocess import normalize_sign, standardize_interaction_table
from src.run import validate_split, encoded_positive_view, encoded_negative_views
from src.util import evaluate_metrics, total_loss

torch.set_num_threads(1)


def configuration(variant="full"):
    return ablation_config(get_args(["--data", "SYNTHETIC", "--hidden_dim", "12", "--identity_dim", "4",
                                    "--intent_dim", "4", "--n_intents", "2", "--n_prototypes", "5",
                                    "--topk", "2", "--dropout", "0", "--layers", "1"]), variant)


def small_graph():
    torch.manual_seed(19)
    pairs = torch.tensor([[d, g] for d in range(6) for g in range(7) if (d + g) % 3 != 0])
    labels = torch.tensor([(d * 3 + g) % 2 for d, g in pairs.tolist()])
    return pairs, labels, torch.randn(6, 5), torch.randn(7, 8)


class SignMappingTests(unittest.TestCase):
    def test_antagonist_not_agonist(self):
        self.assertEqual(normalize_sign("antagonist"), -1)
        self.assertEqual(normalize_sign("partial antagonist"), -1)
        self.assertEqual(normalize_sign("inverse agonist"), -1)
        self.assertEqual(normalize_sign(" AGONIST "), 1)

    def test_conflicting_multi_type_is_not_arbitrarily_positive(self):
        self.assertIsNone(normalize_sign("agonist; inhibitor"))
        self.assertIsNone(normalize_sign('["inhibitor", "activator"]'))
        self.assertEqual(normalize_sign("inhibitor|antagonist"), -1)
        self.assertIsNone(normalize_sign("binding"))

    def test_released_encoding_is_explicit(self):
        mapping = {"increased": 1, "decreased": -1}
        self.assertEqual(normalize_sign("increased", mapping), 1)
        self.assertEqual(normalize_sign("decreased", mapping), -1)
        self.assertIsNone(normalize_sign("inhibitor", mapping))
        self.assertEqual(normalize_sign(0, {"0": -1, "1": 1}), -1)

    def test_pair_conflicts_missing_names_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            source, dest = Path(folder) / "input.csv", Path(folder) / "out.csv"
            pd.DataFrame({"drug": ["D", "D", "X", "X", None, "Q"], "gene": ["G", "G", "Y", "Y", "A", "Z"],
                          "interaction": ["agonist", "inhibitor", "antagonist", "antagonist", "agonist", "binding"]}).to_csv(source, index=False)
            out = standardize_interaction_table(source, dest, sign_col="interaction")
            self.assertEqual(out[["drug", "gene", "sign"]].values.tolist(), [["X", "Y", -1]])
            audit = json.loads(Path(str(dest) + ".audit.json").read_text())
            self.assertEqual(audit["opposite_sign_pairs_excluded"], 1)
            self.assertEqual(audit["exact_duplicates_removed"], 1)

    def test_all_unmapped_writes_empty_valid_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            source, dest = Path(folder) / "in.csv", Path(folder) / "out.csv"
            pd.DataFrame({"drug": ["D"], "gene": ["G"], "sign": ["binding"]}).to_csv(source, index=False)
            out = standardize_interaction_table(source, dest)
            self.assertEqual(list(out.columns), ["drug", "gene", "sign"])
            self.assertTrue(out.empty)


class FeatureAndDataTests(unittest.TestCase):
    def test_feature_ids_reorder_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.npz"
            np.savez(path, ids=np.array(["B", "A"]), features=np.array([[3, 4], [1, 2]]))
            np.testing.assert_array_equal(read_features(path, ["A", "B"]).numpy(), [[1, 2], [3, 4]])

    def test_npy_without_ids_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.npy"
            np.save(path, np.ones((2, 4)))
            with self.assertRaises(ValueError):
                read_features(path, ["A", "B"])

    def test_duplicate_pairs_rejected_before_cv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.csv"
            pd.DataFrame({"drug": ["A", "A"], "gene": ["G", "G"], "sign": [1, -1]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_dataset(path)


class ArchitectureAndLossTests(unittest.TestCase):
    def test_graph_loss_has_effective_encoder_gradients(self):
        result = run_diagnostic()
        self.assertTrue(result["passed"])
        self.assertTrue(all(value > 1e-10 for value in result["encoder_gradient_norms"].values()))

    def test_all_variants_forward_backward_finite(self):
        pairs, labels, df, gf = small_graph()
        pos, neg = build_training_graph(pairs, labels)
        mask = torch.ones(13, dtype=torch.bool)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                args = configuration(variant)
                model = CSIH(args, 6, 7, 5, 8)
                out = model(df, gf, pos, neg, pairs)
                self.assertEqual(out["logits"].shape, (len(pairs), 2))
                v1 = encoded_positive_view(model, df, gf, pos, neg, mask, args)
                v2 = encoded_positive_view(model, df, gf, pos, neg, mask, args)
                negatives = encoded_negative_views(model, df, gf, pos, neg, mask, args) if args.lambda_graph else []
                loss = total_loss(out, labels, v1, v2, args, negatives, model)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_early_fusion_keeps_parameter_count(self):
        full = CSIH(configuration(), 6, 7, 5, 8)
        early = CSIH(configuration("CSIH-EF"), 6, 7, 5, 8)
        self.assertEqual(sum(p.numel() for p in full.parameters()), sum(p.numel() for p in early.parameters()))

    def test_no_hypergraph_means_actual_module_removal(self):
        model = CSIH(configuration("w/o_SIH"), 6, 7, 5, 8)
        self.assertFalse(hasattr(model, "pos_hypergraph"))
        self.assertFalse(any("prototype" in name for name, _ in model.named_parameters()))

    def test_removed_fm_cannot_change_predictions(self):
        pairs, _, df, gf = small_graph()
        pos, neg = build_training_graph(pairs, torch.arange(len(pairs)) % 2)
        model = CSIH(configuration("w/o_FM"), 6, 7, 5, 8).eval()
        first = model(df, gf, pos, neg, pairs)["prob_positive"]
        second = model(df * 99 + 123, gf * -37, pos, neg, pairs)["prob_positive"]
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_removed_id_has_no_embedding_parameters(self):
        model = CSIH(configuration("w/o_ID"), 6, 7, 5, 8)
        self.assertIsNone(model.feature_encoder.drug_identity)
        self.assertIsNone(model.feature_encoder.gene_identity)
        self.assertEqual(model.decoder.mlp[0].in_features, 4 * configuration().hidden_dim)

    def test_symmetric_degree_normalization(self):
        src = torch.tensor([[2.], [4.]])
        result = BipartiteSignAggregator._aggregate(src, torch.tensor([0, 0, 1]), torch.tensor([0, 1, 1]), 2)
        expected = torch.tensor([[2 / np.sqrt(2)], [1 + 4 / np.sqrt(2)]], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_negative_sign_corruption_uses_only_training_pairs(self):
        pairs, labels, _, _ = small_graph()
        pos, neg = build_training_graph(pairs, labels)
        cp, cn, _ = corrupt_training_inputs(pos, neg, torch.ones(13, dtype=torch.bool), 6)
        self.assertEqual(set(map(tuple, pairs.tolist())), set(map(tuple, torch.cat((cp, cn), 1).T.tolist())))
        self.assertFalse(set(map(tuple, cp.T.tolist())) & set(map(tuple, cn.T.tolist())))


class ColdStartTests(unittest.TestCase):
    def test_entity_splits_exclude_test_entities_from_model_selection(self):
        pairs = np.array([[d, g] for d in range(40) for g in range(30)])
        for mode in ("drug", "gene", "double"):
            split = make_cold_start_split(pairs, mode, .2, 71, .2)
            tr, va, te = [pairs[split[k]] for k in ("train_idx", "val_idx", "test_idx")]
            for col, endpoint in ((0, "drug"), (1, "gene")):
                if mode in (endpoint, "double"):
                    self.assertFalse(set(te[:, col]) & (set(tr[:, col]) | set(va[:, col])))
                    self.assertFalse(set(va[:, col]) & set(tr[:, col]))
                else:
                    self.assertTrue(set(te[:, col]).issubset(set(tr[:, col])))

    def _cold_model(self):
        torch.manual_seed(21)
        model = CSIH(configuration(), 6, 7, 5, 8)
        context = torch.tensor([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
        pairs = torch.tensor([[d, g] for d in range(4) for g in range(5) if (d + g) % 3])
        pos, neg = build_training_graph(pairs, torch.arange(len(pairs)) % 2)
        return model, context, pos, neg, torch.randn(6, 5), torch.randn(7, 8)

    def test_held_out_features_do_not_affect_training_representations(self):
        model, ctx, p, n, df, gf = self._cold_model()
        model.eval()
        before = model.encode(df, gf, p, n, context_mask=ctx, include_unseen=False)["all"]
        df[4:] = 1000
        gf[5:] = -1000
        after = model.encode(df, gf, p, n, context_mask=ctx, include_unseen=False)["all"]
        torch.testing.assert_close(before, after, atol=0, rtol=0)

    def test_held_out_features_and_ids_receive_no_training_gradients(self):
        model, ctx, p, n, df, gf = self._cold_model()
        df, gf = df.requires_grad_(), gf.requires_grad_()
        model.train()
        model.encode(df, gf, p, n, context_mask=ctx)["all"].sum().backward()
        self.assertEqual(float(df.grad[4:].abs().sum()), 0)
        self.assertEqual(float(gf.grad[5:].abs().sum()), 0)
        self.assertEqual(float(model.feature_encoder.drug_identity.weight.grad[4:].abs().sum()), 0)

    def test_other_unseen_queries_do_not_change_a_query_prediction(self):
        model, ctx, p, n, df, gf = self._cold_model()
        model.eval()
        pairs = torch.tensor([[4, 5]])
        before = model(df, gf, p, n, pairs, context_mask=ctx)["prob_positive"]
        df[5] = 1000
        gf[6] = -2000
        after = model(df, gf, p, n, pairs, context_mask=ctx)["prob_positive"]
        torch.testing.assert_close(before, after, atol=1e-7, rtol=0)

    def test_held_out_graph_edge_is_rejected(self):
        model, ctx, p, n, df, gf = self._cold_model()
        bad = torch.cat((p, torch.tensor([[4], [1]])), 1)
        with self.assertRaises(ValueError):
            model.encode(df, gf, bad, n, context_mask=ctx)


class AuditAndStatisticsTests(unittest.TestCase):
    def test_condition_conflicts_and_reconstruction_are_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            records = []
            for drug, gene, cell, z in [("D1", "G1", "A", 4), ("D1", "G1", "A", 2),
                                       ("D1", "G1", "B", -4), ("D1", "G2", "A", 3),
                                       ("D1", "G2", "B", 1), ("D2", "G3", "A", 0),
                                       ("D3", "G4", "A", np.nan)]:
                records.append(dict(drug=drug, gene=gene, cell_line=cell, dose=1, dose_unit="uM",
                                    duration=6, duration_unit="h", z_score=z, signature_id=f"SIG_{len(records)}"))
            pd.DataFrame(records).to_csv(folder / "expression.csv", index=False)
            pd.DataFrame(dict(drug=["D1", "D1", "D2", "D3"], gene=["G1", "G2", "G3", "G4"],
                              sign=[-1, 1, 1, -1])).to_csv(folder / "benchmark.csv", index=False)
            before = (folder / "benchmark.csv").read_bytes()
            audit = audit_conditions(folder / "expression.csv", folder / "benchmark.csv", folder / "audit", label_threshold=0)
            self.assertEqual(audit["candidate_pairs_with_condition_metadata"], 4)
            self.assertEqual(audit["pairs_without_usable_condition_scores"], 1)
            self.assertEqual(audit["discordant_pairs"], 1)
            self.assertEqual(audit["retention"]["median"]["pairs"], 2)
            self.assertEqual(audit["retention"]["median_consistent"]["pairs"], 1)
            self.assertEqual((folder / "benchmark.csv").read_bytes(), before)

    def test_frequency_baseline_fallbacks_and_constant_metrics(self):
        baseline = SignFrequencyBaseline().fit([[0, 0], [0, 1], [1, 1]], [1, 1, 0])
        result = baseline.predict([[0, 99], [99, 1], [99, 99]], "Drug-Gene-Prior")
        np.testing.assert_allclose(result, [1, .5, 2 / 3])
        metrics = evaluate_metrics([0, 0, 1, 0], [.25] * 4)
        self.assertEqual(metrics, {"AUC": .5, "AUPR": .25, "Binary-F1": 0.0})

    def test_holm_matches_nine_reported_raw_p_values(self):
        p = [.0064, .0352, .0026, .0407, .0006, .0018, .1176, .0216, .0232]
        np.testing.assert_allclose(holm_adjust(p), [.0384, .108, .0182, .108, .0054, .0144, .1176, .108, .108])

    def test_statistics_require_matched_seeds(self):
        rows = []
        for method, seeds in (("full", [1, 2, 3]), ("Other", [1, 2])):
            for seed in seeds:
                rows.append(dict(dataset="SYNTHETIC", protocol="random", method=method, seed=seed,
                                 AUC=.7, AUPR=.6, **{"Binary-F1": .5}))
        with self.assertRaises(ValueError):
            compare_seed_scores(pd.DataFrame(rows), "full", ["Other"])


if __name__ == "__main__":
    unittest.main()
