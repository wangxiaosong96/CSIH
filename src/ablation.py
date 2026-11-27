"""Explicit ablations; no single-prototype approximation of module removal."""
from copy import deepcopy

VARIANTS = ("full", "CSIH-EF", "w/o_FM", "w/o_ID", "w/o_SIH", "w/o_MI",
            "w/o_NCL", "w/o_GCL", "w/o_DrugFeat", "w/o_GeneFeat", "w/o_IDR")


def ablation_config(args, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    cfg = deepcopy(args)
    cfg.variant = variant
    cfg.use_drug_features = variant not in ("w/o_FM", "w/o_DrugFeat")
    cfg.use_gene_features = variant not in ("w/o_FM", "w/o_GeneFeat")
    cfg.use_id_embeddings = variant != "w/o_ID"
    cfg.use_hypergraph = variant != "w/o_SIH"
    cfg.early_fusion = variant == "CSIH-EF"
    if variant == "w/o_MI":
        cfg.n_intents = 1
    if variant == "w/o_NCL":
        cfg.lambda_node = 0.0
    if variant == "w/o_GCL":
        cfg.lambda_graph = 0.0
    if variant == "w/o_IDR" or variant == "w/o_SIH":
        cfg.lambda_decor = 0.0
    return cfg
