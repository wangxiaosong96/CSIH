"""Ablation configuration helpers corresponding to the manuscript."""
from copy import deepcopy


def ablation_config(args, variant):
    cfg = deepcopy(args)

    if variant == "full":
        return cfg
    if variant == "w/o_MI":
        cfg.n_intents = 1
    elif variant == "w/o_NCL":
        cfg.lambda_node = 0.0
    elif variant == "w/o_GCL":
        cfg.lambda_graph = 0.0
    elif variant == "w/o_IDR":
        cfg.lambda_decor = 0.0
    elif variant == "w/o_SIH":
        cfg.n_intents = 1
        cfg.n_prototypes = 1
        cfg.topk = 1
    else:
        raise ValueError(
            "Supported: full, w/o_MI, w/o_NCL, w/o_GCL, w/o_IDR, w/o_SIH. "
            "CSIH-EF and w/o_FM require architectural/input changes and should "
            "be implemented explicitly rather than silently approximated."
        )
    return cfg
