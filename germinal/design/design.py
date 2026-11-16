"""Germinal Design Module

This module provides the core functionality for protein binder design using ColabDesign's
AlphaFold2-based hallucination approach. It includes the main design function and various
loss functions for optimizing protein structure and binding properties.

The module implements:
- Binder hallucination using AF2 models
- Various structural loss functions (helix, beta strand/sheet, radius of gyration)
- Interface optimization losses (pTM, termini distance)
- Trajectory logging and visualization

Two design styles share one implementation, selected by the ``stages`` argument
of :func:`germinal_design`:

- ``STAGES_FULL``   -- logits, then softmax, then PSSM semigreedy.
- ``STAGES_LOGITS`` -- logits only, the schedule the multi-target recycling
  runner uses.

Key Functions:
    germinal_design: Main function for conducting binder hallucination
    get_best_plddt: Extract confidence metrics from best model
    get_best_pae_ipae: Extract PAE metrics from best model
    add_*_loss: Various structural and interface loss functions
    log_trajectory: Save trajectory metrics to CSV
    plot_trajectory: Generate trajectory plots

"""

import os
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import jax
import jax.numpy as jnp
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins
from germinal.utils.utils import hotspot_residues, calculate_clash_score
from germinal.utils.io import IO

from germinal.utils.log import get_logger

log = get_logger(__name__)


#: Where the AlphaFold-Multimer parameters must live. ColabDesign reads
#: <data_dir>/params/params_model_*.npz, so data_dir is the directory holding
#: params/ -- the repository root. Deliberately not configurable: a second
#: location to get wrong, when a symlink into params/ does the same job.
AF_PARAMS_DATA_DIR = str(Path(__file__).resolve().parents[2])


# The two design schedules. "logits" is in both: it is the stage that actually
# produces a sequence, and the later two only refine it.
STAGES_FULL = ("logits", "soft", "greedy")
STAGES_LOGITS = ("logits",)


# One compiled AF model, reused across a run's trajectories. Keyed on every
# parameter it is built with, since one function serves both design styles.
_MODEL_CACHE = {}


def germinal_design(
    design_name: str,
    run_settings: dict,
    target_settings: dict,
    io: IO,
    seed: int = None,
    stages: tuple = STAGES_FULL,
    model_pdb_path: str = None,
):
    """
    Conduct binder hallucination with ColabDesign AF2 model.

    Args:
        design_name: Unique name for this design trajectory
        run_settings: Dictionary containing all run configuration parameters
        target_settings: Dictionary containing target-specific parameters
        io: IO handler for file operations
        seed: Seed for the design
        stages: Which optimisation stages to run -- STAGES_FULL or
            STAGES_LOGITS. See the module docstring.
        model_pdb_path: Where to write the hallucinated PDB. Defaults to the
            standard location under the run's trajectories directory; the
            runner passes an explicit path.

    Returns:
        Configured AF model instance with hallucination results. Failure
        charges for failure_counts.csv are collected on the returned model as
        ``_hallucination_failures``; the caller submits them via
        ``io.update_failures``. They are returned rather than written here
        because the caller records a trajectory row in the same breath, and
        splitting the write between here and there made the counters hard to
        follow.
    """
    # Extract configuration parameters from input dictionaries
    starting_pdb = run_settings["starting_pdb_complex"]
    chain = target_settings["target_chain"]
    target_hotspot_residues = target_settings.get("target_hotspots", "")
    design_models = run_settings.get("design_models", [0,1,2,3,4])
    num_targets = target_settings.get("num_targets", 1)

    # Unpack individual parameters from run_settings
    pos = run_settings.get("cdr_positions")
    cdr_lengths = run_settings.get("cdr_lengths")
    fw_lengths = run_settings.get("fw_lengths")
    clear_best = run_settings.get("clear_best", True)
    clear_best_greedy = run_settings.get("clear_best_greedy", False)
    bias_redesign = run_settings.get("bias_redesign", 10)
    binder_chain = target_settings.get("binder_chain", "B")
    rm_binder_seq = run_settings.get("rm_binder_seq", True)
    rm_binder_sc = run_settings.get("rm_binder_sc", True)
    rm_binder = run_settings.get("rm_binder", False)
    learning_rate = run_settings.get("learning_rate", 0.1)
    optimizer = run_settings.get("optimizer", "sgd")
    num_models = run_settings.get("num_models", 1)
    recycle_mode = run_settings.get("recycle_mode", "last")

    length = sum(fw_lengths)+sum(cdr_lengths)

    use_pos_distance = run_settings.get("use_pos_distance", True)
    iglm_grad_merge_method = run_settings.get("iglm_grad_merge_method", "pcgrad")
    iglm_scale = run_settings.get("iglm_scale", [0.0, 0.2, 0.4, 1.0])
    iglm_temp = run_settings.get("iglm_temp", 0.6) 
    vh_len = run_settings.get("vh_len", None)
    vh_first = run_settings.get("vh_first", True)
    vl_len = run_settings.get("vl_len", None)
    iglm_species = run_settings.get("iglm_species", "[HUMAN]")
    # Which antibody language model steers the gradient. IgLM is the only
    # supported value; prep_iglm rejects anything else.
    ablm_model = run_settings.get("ablm_model", "iglm")
    save_filters = {
        "plddt": run_settings.get("plddt_threshold", 0.84),
        "i_ptm": run_settings.get("i_ptm_threshold", 0.65),
        "i_pae": run_settings.get("i_pae_threshold", 0.3),
    }
    seq_init_mode = run_settings.get("seq_init_mode")
    starting_binder_seq = run_settings.get("starting_binder_seq", None)
    normalize_gradient = run_settings.get("normalize_gradient", True)
    linear_lr_annealing = run_settings.get("linear_lr_annealing", False)
    min_lr_scale = run_settings.get("min_lr_scale", 0.01)
    if model_pdb_path is None:
        model_pdb_path = str(io.layout.trajectories / f"structures/{design_name}.pdb")

    protocol = run_settings.get("protocol", "multi_binder")

    # Initialize binder hallucination model only once to avoid recompilation
    cache_key = (
        protocol,
        AF_PARAMS_DATA_DIR,
        run_settings["use_multimer_design"],
        run_settings["num_recycles_design"],
        recycle_mode,
        iglm_grad_merge_method,
        tuple(iglm_scale),
        normalize_gradient,
        linear_lr_annealing,
        min_lr_scale,
        length,
        # The add_*_loss calls mutate the model after construction, so the
        # loss set is part of the cache key: a jitted model cannot drop one.
        run_settings.get("use_rg_loss", True),
        run_settings.get("use_i_ptm_loss", False),
        run_settings.get("use_termini_distance_loss", False),
        run_settings.get("use_helix_loss", True),
        run_settings.get("use_beta_loss", True),
        bool(pos is not None and pos != ""),
    )
    if _MODEL_CACHE.get("key") != cache_key:
        # Clear GPU memory before first compilation
        clear_mem()
        log.info(f"Using gradient merging method: {iglm_grad_merge_method}")
        _MODEL_CACHE["key"] = cache_key
        _MODEL_CACHE["model"] = mk_afdesign_model(
            protocol=protocol,
            debug=False,
            data_dir=AF_PARAMS_DATA_DIR,
            use_multimer=run_settings["use_multimer_design"],
            num_recycles=run_settings["num_recycles_design"],
            best_metric="loss",
            recycle_mode=recycle_mode,
            # colabdesign's own parameter name; only the Germinal-side config
            # key is iglm_grad_merge_method.
            grad_merge_method={
                "scale": iglm_grad_merge_method == "scale",
                "pcgrad": iglm_grad_merge_method == "pcgrad",
            },
            iglm_scale=iglm_scale,
            norm_seq_grad=normalize_gradient,
            linear_lr_annealing=linear_lr_annealing,
            min_lr_scale=min_lr_scale,
        )
    af_model = _MODEL_CACHE["model"]

    # Failure charges for failure_counts.csv. Reset per call, since af_model
    # is cached across trajectories.
    hallucination_failures = {}
    af_model._hallucination_failures = hallucination_failures

    # Validate hotspot residue specification
    if target_hotspot_residues == "":
        target_hotspot_residues = None

    seq_entropy_threshold = run_settings.get("seq_entropy_threshold", 0.1)
    if seq_entropy_threshold == 0:
        seq_entropy_threshold = None
        log.info("Sequence entropy threshold set to 0, disabling filter")

    af_model.prep_inputs(
        pdb_filename=starting_pdb,
        chain=chain,
        binder_len=length,
        hotspot=target_hotspot_residues,
        seed=seed,
        rm_target_seq=run_settings.get("rm_template_seq", True),
        rm_target_sc=run_settings.get("rm_template_sc", True),
        binder_chain=binder_chain,
        rm_binder=rm_binder,
        rm_binder_seq=rm_binder_seq,
        rm_binder_sc=rm_binder_sc,
        pos=pos,
        cdr_lengths=cdr_lengths,
        bias_redesign=bias_redesign,
        learning_rate=learning_rate,
        optimizer=optimizer,
        iglm_temp=iglm_temp,
        iglm_species=iglm_species,
        ablm_model=ablm_model,
        vl_len=vl_len,
        vh_first=vh_first,
        vh_len=vh_len,
        use_pos_distance=use_pos_distance,
        rm_template_ic=run_settings.get("rm_template_ic", False),
        rm_aa=run_settings.get("omit_AAs", ""),
        starting_binder_seq=starting_binder_seq,
        mode=seq_init_mode,
        lens={
            "fw": run_settings.get("fw_lengths"),
            "cdrs": cdr_lengths,
        },
        multi_grad_merge_method = run_settings.get("multi_grad_merge_method", "sum"),
        is_recover = run_settings.get("is_recover", False),
        recover_bias = run_settings.get("recover_bias", 0)
    )
    
    # Configure loss function weights based on specified settings
    af_model.opt["weights"].update(
        {
            "pae": run_settings["weights_pae_intra"],
            "i_plddt": run_settings["weights_i_plddt"],
            "plddt": run_settings["weights_plddt"],
            "i_pae": run_settings["weights_pae_inter"],
            "con": run_settings["weights_con_intra"],
            "i_con": run_settings["weights_con_inter"],
            "dgram_cce": run_settings.get("dgram_cce", 0.01),
        }
    )

    # Configure intramolecular (con) and intermolecular (i_con) contact definitions
    af_model.opt["con"].update(
        {
            "num": run_settings["intra_contact_number"],
            "cutoff": run_settings["intra_contact_distance"],
            "binary": False,
            "seqsep": 9,
        }
    )
    af_model.opt["i_con"].update(
        {
            "num": run_settings["inter_contact_number"],
            "cutoff": run_settings["inter_contact_distance"],
            "binary": False,
            "framework_contact_loss": run_settings["framework_contact_loss"],
            "framework_contact_offset": run_settings["framework_contact_offset"],
        }
    )

    # Drop losses registered by a previous trajectory first: af_model is
    # cached and each add_*_loss appends unconditionally.
    af_model._callbacks["model"]["loss"].clear()

    if run_settings.get("use_rg_loss", True):
        # Apply radius of gyration loss to control protein compactness
        add_rg_loss(af_model, run_settings.get("weights_rg", 0.1))

    if run_settings.get("use_i_ptm_loss", False):
        # Apply interface predicted Template Modeling score loss
        add_i_ptm_loss(af_model, run_settings.get("weights_iptm", 0.75))

    if run_settings.get("use_termini_distance_loss", False):
        # Apply N- and C-terminus distance constraint loss
        add_termini_distance_loss(
            af_model, run_settings.get("weights_termini_loss", 0.1)
        )

    # Apply helicity loss to promote alpha-helical secondary structure
    if (pos is not None and pos != "") and run_settings.get("use_helix_loss", True):
        weights_helix = run_settings.get("weights_helix", 0.1)
        add_helix_loss(af_model, weights_helix)

    if (pos is not None and pos != "") and run_settings.get("use_beta_loss", True):
        beta_strand_weight = run_settings.get("weights_beta", 0.1)
        add_beta_strand_loss(af_model, beta_strand_weight)

    # Calculate the number of mutations for greedy optimization based on protein length
    greedy_tries = math.ceil(length * run_settings.get("search_mutation_rate", 0.05))

    # Perform initial logits optimization to prescreen trajectory viability
    if stages == STAGES_LOGITS:
        log.info("Logits-only design")
    else:
        log.info("Stage 1: Phase Logits")
    fail_confidence = False

    af_model.design_logits(
        iters=run_settings["logits_steps"],
        soft=0,
        e_soft=1,
        models=design_models,
        num_models=num_models,
        sample_models=run_settings.get("sample_models", True),
        save_best=True,
    )

    # Evaluate confidence metrics of the best iteration based on lowest loss value
    initial_plddt, initial_iptm = get_best_plddt(af_model, length)
    initial_pae, initial_ipae = get_best_pae_ipae(af_model, length)

    # The gate below only decides whether to go on to softmax, so a
    # logits-only run skips it and leaves fail_confidence False.
    run_later_stages = ("soft" in stages) or ("greedy" in stages)

    # Proceed with optimization if initial trajectory meets confidence thresholds
    if not run_later_stages:
        pass
    elif initial_plddt > save_filters["plddt"] and initial_iptm >= save_filters["i_ptm"]:
        log.info(
            f"Initial trajectory pLDDT/iPTM good, continuing: "
            f"{initial_plddt} {initial_iptm}"
        )
        # perform softmax trajectory design
        if "soft" in stages and run_settings.get("softmax_steps", 35) > 0:
            if (
                clear_best
                and seq_entropy_threshold is not None
                and af_model._tmp["best"]["mean_soft_pseudo"] < seq_entropy_threshold
            ):
                log.info(
                    f"Clearing best model due to low sequence entropy: "
                    f"{af_model._tmp['best']['mean_soft_pseudo']}"
                )
                af_model.clear_best()
            log.info("\n\n\nPhase 2: Softmax Optimisation")
            af_model.design_soft(
                run_settings.get("softmax_steps", 35),
                e_temp=1e-2,
                models=design_models,
                num_models=num_models,
                sample_models=run_settings.get("sample_models", True),
                ramp_recycles=False,
                save_best=True,
                save_filters=save_filters,
                seq_entropy_threshold=seq_entropy_threshold,
            )
            softmax_plddt, softmax_iptm = get_best_plddt(af_model, length)
            softmax_pae, softmax_ipae = get_best_pae_ipae(af_model, length)
        else:
            softmax_plddt = initial_plddt
            softmax_iptm = initial_iptm
            softmax_ipae = initial_ipae

        # perform one hot encoding
        if (
            softmax_plddt > save_filters["plddt"]
            and softmax_iptm >= save_filters["i_ptm"]
            and softmax_ipae < save_filters["i_pae"]
            # seq_entropy_threshold is None when the config sets it to 0, which
            # disables the entropy gate; colabdesign applies the same guard.
            and (
                seq_entropy_threshold is None
                or af_model._tmp["best"]["mean_soft_pseudo"] >= seq_entropy_threshold
            )
        ):
            log.info(
                f"Softmax trajectory pLDDT good, continuing (plddt/iptm/ipae): "
                f"{softmax_plddt} {softmax_iptm} {softmax_ipae} "
                f"{af_model._tmp['best']['mean_soft_pseudo']}"
            )
            if "greedy" in stages and run_settings.get("search_steps", 10) > 0:
                log.info("\n\n\nPhase 3: PSSM Semigreedy Optimisation")
                best_for_greedy = True
                af_model.design_pssm_semigreedy(
                    soft_iters=0,
                    hard_iters=run_settings.get("search_steps", 10),
                    tries=greedy_tries,
                    models=design_models,
                    save_filters=save_filters,
                    num_models=1,
                    sample_models=run_settings.get("sample_models", True),
                    ramp_models=False,
                    save_best=True,
                    get_best=best_for_greedy,
                    clear_best=clear_best_greedy
                )

        else:
            # Same conjunction problem as the logits gate, plus the entropy
            # clause, which is a distinct failure and not a confidence one.
            softmax_failures = {}
            if not softmax_plddt > save_filters["plddt"]:
                softmax_failures["hallucination_softmax_plddt"] = 1
            if not softmax_iptm >= save_filters["i_ptm"]:
                softmax_failures["hallucination_softmax_i_ptm"] = 1
            if not softmax_ipae < save_filters["i_pae"]:
                softmax_failures["hallucination_softmax_i_pae"] = 1
            if (
                seq_entropy_threshold is not None
                and not af_model._tmp["best"]["mean_soft_pseudo"]
                >= seq_entropy_threshold
            ):
                softmax_failures["hallucination_softmax_seq_entropy"] = 1
            if softmax_failures:
                hallucination_failures.update(softmax_failures)
            log.info(
                f"Softmax trajectory metrics too low to continue: "
                f"{softmax_plddt}/{softmax_iptm}/{softmax_ipae}"
            )
            fail_confidence = True
    else:
        # The gate is a conjunction of pLDDT and iPTM, so charge whichever
        # fell short. It does not test i_pae, which is never charged here.
        logits_failures = {}
        if not initial_plddt > save_filters["plddt"]:
            logits_failures["hallucination_logits_plddt"] = 1
        if not initial_iptm >= save_filters["i_ptm"]:
            logits_failures["hallucination_logits_i_ptm"] = 1
        if logits_failures:
            hallucination_failures.update(logits_failures)
        log.info(
            f"Initial trajectory metrics too low to continue: "
            f"{initial_plddt}/{initial_iptm}/{initial_ipae}"
        )
        fail_confidence = True

    ### save trajectory PDB
    final_plddt, final_iptm = get_best_plddt(af_model, length)
    final_pae, final_ipae = get_best_pae_ipae(af_model, length)
    log.info(
        f"Final pLDDT/iPTM/iPAE for design {design_name}: "
        f"{final_plddt} {final_iptm} {final_ipae}"
    )
    af_model.save_pdb(model_pdb_path, save_all=False)
    af_model.aux["log"]["terminate"] = ""

    # let's check whether the trajectory is worth optimising by checking confidence, clashes, and contacts
    # for a multi-target run, we take the maximum number of clashes across targets
    ca_clashes = calculate_clash_score(
        model_pdb_path,
        run_settings["clash_threshold"],
        only_ca=True,
        num_targets=num_targets,
    )

    if ca_clashes > 0:
        af_model.aux["log"]["terminate"] = "Clashing"
        if not fail_confidence:
            hallucination_failures["hallucination_clashes"] = 1
        log.info("Severe clashes detected, skipping analysis")
        log.info("")
    else:
        # check if low quality prediction
        if (
            final_plddt < save_filters["plddt"]
            or final_iptm < save_filters["i_ptm"]
            or final_ipae >= save_filters["i_pae"]
            or fail_confidence
        ):
            af_model.aux["log"]["terminate"] = "LowConfidence"
            if not fail_confidence:
                # A disjunction, so charge each metric that fell short.
                final_failures = {}
                if final_plddt < save_filters["plddt"]:
                    final_failures["hallucination_final_plddt"] = 1
                if final_iptm < save_filters["i_ptm"]:
                    final_failures["hallucination_final_i_ptm"] = 1
                if final_ipae >= save_filters["i_pae"]:
                    final_failures["hallucination_final_i_pae"] = 1
                if final_failures:
                    hallucination_failures.update(final_failures)
            log.info("Trajectory final confidence low, skipping analysis")
            log.info("")
        else:
            # does it have enough contacts to consider?
            binder_contacts = hotspot_residues(
                model_pdb_path,
                binder_chain=binder_chain,
                target_chain=chain,
                num_targets=num_targets,
            )
            binder_contacts_n = len(binder_contacts.items())

            # if less than 3 contacts then protein is floating above and is not binder
            # for a multi-target run, we take the minimum number of contacts across targets
            if binder_contacts_n < 3:
                af_model.aux["log"]["terminate"] = "LowConfidence"
                if not fail_confidence:
                    hallucination_failures["hallucination_contacts"] = 1
                log.info("Too few contacts at the interface, skipping analysis")
                log.info("")
            else:
                af_model.aux["log"]["terminate"] = ""
                log.info(
                    f"Trajectory successful, final pLDDT/iPTM/iPAE: "
                    f"{final_plddt} {final_iptm} {final_ipae}"
                )

    ### get the sampled sequence for plotting
    af_model.get_seqs()
    if run_settings.get("save_design_trajectory_plots", False):
        log_trajectory(af_model, design_name, io)
        plot_trajectory(af_model, design_name, io)

    return af_model


def get_best_plddt(af_model, length):
    """Extract confidence metrics from the best model iteration.
    
    Calculates the predicted Local Distance Difference Test (pLDDT) and
    interface predicted Template Modeling score (iPTM) for the binder region
    of the best model according to the lowest loss value.
    
    Args:
        af_model: ColabDesign AF model instance containing optimization results
        length (int): Length of the binder sequence for metric calculation
        
    Returns:
        tuple: (plddt, iptm) where:
            - plddt (float): Mean pLDDT score for the binder region (0-1 scale)
            - iptm (float): Mean interface pTM score (0-1 scale)
    """

    # for multi target designs, we need to average the plddt scores for each target
    if af_model.protocol == 'multi_binder':
        plddt = round(np.mean([np.mean(tgt_plddt[-length:]) for tgt_plddt in af_model._tmp["best"]["aux"]["plddt"]]), 3)
        iptm = round(np.mean([np.mean(tgt_conf["i_ptm"][-length:]) for tgt_conf in af_model._tmp["best"]["aux"]["all"]]), 3)
    else:
        plddt = round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]), 3)
        iptm = round(np.mean(af_model._tmp["best"]["aux"]["all"]["i_ptm"][-length:]), 3)

    return plddt, iptm


def get_best_pae_ipae(af_model, length):
    """Extract Predicted Aligned Error metrics from the best model iteration.
    
    Retrieves the Predicted Aligned Error (PAE) and interface PAE (iPAE) values
    from the best model iteration for structure quality assessment.
    
    Args:
        af_model: ColabDesign AF model instance containing optimization results
        length (int): Length parameter (currently unused but maintained for API consistency)
        
    Returns:
        tuple: (pae, ipae) where:
            - pae (float): Predicted Aligned Error for the overall structure
            - ipae (float): Interface Predicted Aligned Error between binder and target
    """
    pae = af_model._tmp["best"]["aux"]["log"]["pae"]
    ipae = af_model._tmp["best"]["aux"]["log"]["i_pae"]
    return pae, ipae


def add_rg_loss(self, weight=0.1):
    """Add radius of gyration loss function to control protein compactness.
    
    Implements a radius of gyration constraint to encourage compact protein
    structures. The loss penalizes structures that deviate from the expected
    radius of gyration based on the protein length.
    
    Args:
        self: ColabDesign AF model instance
        weight (float, optional): Weight for the radius of gyration loss term. Defaults to 0.1.
        
    Note:
        The theoretical radius of gyration is calculated using the empirical formula:
        rg_th = 2.38 * N^0.365, where N is the number of residues.
    """

    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len :]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight


def add_i_ptm_loss(self, weight=0.1):
    """Add interface predicted Template Modeling score loss function.
    
    Implements an interface pTM loss to optimize the predicted confidence
    of the protein-protein interface. This loss encourages high-confidence
    binding interfaces by penalizing low interface pTM scores.
    
    Args:
        self: ColabDesign AF model instance
        weight (float, optional): Weight for the interface pTM loss term. Defaults to 0.1.
        
    Note:
        The loss is computed as (1 - interface_pTM) to minimize when interface
        confidence is high.
    """

    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}

    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight


def add_helix_loss(self, weight=0):
    """Add helical secondary structure loss function.
    
    Implements a loss function that promotes alpha-helical secondary structure
    in specified regions of the protein. The loss is based on distance constraints
    typical of alpha-helical geometry (i, i+3 contacts).
    
    Args:
        self: ColabDesign AF model instance
        weight (float, optional): Weight for the helix loss term. Defaults to 0.
        
    Note:
        The loss uses distance cutoffs of 2.0-6.2 Å to identify helical contacts
        and applies the constraint either globally or to specific positions
        defined in self.opt['pos'].
    """

    def _get_con_loss_h(dgram, dgram_bins, cutoff_lower=None, cutoff=None, binary=True):
        """dgram to contacts"""
        if cutoff is None:
            cutoff = dgram_bins[-1]
        if cutoff_lower is None:
            cutoff_lower = dgram_bins[0]
        # bins = (dgram_bins < cutoff) & (dgram_bins > cutoff_lower)
        bins = jnp.logical_or(dgram_bins > cutoff, dgram_bins < cutoff_lower)
        px = jax.nn.softmax(dgram)
        px_ = jax.nn.softmax(dgram - 1e7 * (1 - bins))
        # binary/categorical cross-entropy
        con_loss_cat_ent = -(px_ * jax.nn.log_softmax(dgram)).sum(-1)
        con_loss_bin_ent = -jnp.log((bins * px + 1e-8).sum(-1))
        return jnp.where(binary, con_loss_bin_ent, con_loss_cat_ent)

    def binder_helicity(inputs, outputs):
        if "offset" in inputs:
            offset = inputs["offset"]
        else:
            idx = inputs["residue_index"].flatten()
            offset = idx[:, None] - idx[None, :]

        # define distogram
        dgram = outputs["distogram"]["logits"]  # L, L, 64
        dgram_bins = get_dgram_bins(outputs)  # 64

        if "pos" in self.opt:
            # Create a 1D mask with zeros then set indices from self.opt['pos'] to 1
            mask_2d_ = jnp.concatenate(
                [jnp.zeros(self._target_len), jnp.zeros(self._binder_len)]
            )
            mask_2d_ = mask_2d_.at[self.opt["pos"]].set(1)
            mask_2d = jnp.outer(mask_2d_, mask_2d_)
        else:
            mask_2d = jnp.outer(
                jnp.concatenate(
                    [jnp.zeros(self._target_len), jnp.ones(self._binder_len)]
                ),
                jnp.concatenate(
                    [jnp.zeros(self._target_len), jnp.ones(self._binder_len)]
                ),
            )

        x = _get_con_loss_h(dgram, dgram_bins, cutoff_lower=2, cutoff=6.2, binary=True)
        if offset is None:
            if mask_2d is None:
                helix_loss = jnp.diagonal(x, 3).mean()
            else:
                helix_loss = jnp.diagonal(x * mask_2d, 3).sum() + (
                    jnp.diagonal(mask_2d, 3).sum() + 1e-8
                )
        else:
            mask = offset == 3
            if mask_2d is not None:
                mask = jnp.where(mask_2d, mask, 0)
            helix_loss = jnp.where(mask, x, 0.0).sum() / (mask.sum() + 1e-8)

        return {"helix": helix_loss}

    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight


def add_beta_strand_loss(self, weight=0):
    """Add beta strand secondary structure loss function.
    
    Implements a loss function that promotes beta strand secondary structure
    in specified regions of the protein. The loss is based on distance constraints
    typical of beta strand geometry.
    
    Args:
        self: ColabDesign AF model instance
        weight (float, optional): Weight for the beta strand loss term. Defaults to 0.
        
    Note:
        The loss uses distance cutoffs of 9.75-11.5 Å to identify beta strand
        contacts and applies the constraint to positions defined in self.opt['pos'].
    """

    def _get_con_loss_beta(
        dgram, dgram_bins, cutoff_lower=None, cutoff=None, binary=True
    ):
        """dgram to contacts"""
        if cutoff is None:
            cutoff = dgram_bins[-1]
        if cutoff_lower is None:
            cutoff_lower = dgram_bins[0]
        bins = jnp.logical_or(dgram_bins > cutoff, dgram_bins < cutoff_lower)

        px = jax.nn.softmax(dgram)
        px_ = jax.nn.softmax(dgram - 1e7 * (1 - bins))
        # binary/cateogorical cross-entropy
        con_loss_cat_ent = -(px_ * jax.nn.log_softmax(dgram)).sum(-1)
        con_loss_bin_ent = -jnp.log((bins * px + 1e-8).sum(-1))
        return jnp.where(binary, con_loss_bin_ent, con_loss_cat_ent)

    def beta_strand_loss(inputs, outputs):
        binder_len = self._binder_len
        target_len = self._target_len

        # define distogram
        dgram = outputs["distogram"]["logits"]  # L,L,64
        dgram_bins = get_dgram_bins(outputs)  # 64

        x = _get_con_loss_beta(
            dgram, dgram_bins, cutoff_lower=9.75, cutoff=11.5, binary=True
        )  # 9.5 and 11.5

        pos = jnp.array(self.opt["pos"])
        mask_2d = jnp.zeros(target_len + binder_len)
        mask_2d = mask_2d.at[pos - 1].set(1).at[pos + 1].set(1)
        mask_2d = jnp.outer(mask_2d, mask_2d)

        loss_array = jnp.sort(jnp.diagonal(mask_2d * x, 3), descending=True)
        top_k_mask = jnp.arange(loss_array.size) < (pos.size // 3)
        beta_loss = jnp.where(top_k_mask, loss_array, 0).sum(-1) / (
            top_k_mask.sum() + 1e-8
        )

        return {"beta_strand": beta_loss}

    self._callbacks["model"]["loss"].append(beta_strand_loss)
    self.opt["weights"]["beta_strand"] = weight


def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    """Add N- and C-terminus distance constraint loss function.
    
    Implements a loss function that constrains the distance between the N- and
    C-termini of the binder protein. This can be useful for promoting compact
    structures or specific geometric arrangements.
    
    Args:
        self: ColabDesign AF model instance
        weight (float, optional): Weight for the termini distance loss term. Defaults to 0.1.
        threshold_distance (float, optional): Target distance between termini in Angstroms. Defaults to 7.0.
        
    Note:
        The loss uses ELU activation followed by ReLU to ensure non-negative values
        and smooth gradients. Only applies to the binder region of the protein.
    """

    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len :]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight


def log_trajectory(af_model, design_name, io):
    """Log design trajectory metrics to CSV file for analysis.
    
    Extracts and saves various optimization metrics from the design trajectory
    to a CSV file for subsequent analysis and visualization. Handles variable-length
    trajectories by padding missing values appropriately.
    
    Args:
        af_model: ColabDesign AF model instance containing trajectory data
        design_name (str): Unique identifier for this design trajectory
        io (IO): IO handler instance for file path management
        
    Note:
        Saves metrics including loss, confidence scores (pLDDT, pTM), contact scores,
        PAE values, secondary structure losses, and language model likelihood.
    """
    metrics_to_log = [
        "loss",
        "plddt",
        "ptm",
        "i_ptm",
        "con",
        "i_con",
        "pae",
        "i_pae",
        "helix",
        "beta_strand",
        "iglm_ll",
        "i_plddt",
    ]

    all_metrics = {}
    max_len = 0

    # First collect all available metrics
    for metric in metrics_to_log:
        if metric in af_model.aux["log"]:
            loss = af_model.get_loss(metric)
            all_metrics[metric] = loss
            max_len = max(max_len, len(loss))

    # Save all metrics to CSV
    if all_metrics:
        plots_dir = os.path.join(io.layout.trajectories, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        csv_path = os.path.join(plots_dir, f"{design_name}_metrics.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["iteration"] + list(all_metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(max_len):
                row = {"iteration": i + 1}
                for metric, values in all_metrics.items():
                    if i < len(values):
                        row[metric] = values[i]
                writer.writerow(row)


def plot_trajectory(af_model, design_name, io):
    """Generate and save trajectory loss plots for design analysis.
    
    Creates individual plots for key optimization metrics throughout the design
    trajectory. Saves both PNG images and raw data files for further analysis.
    
    Args:
        af_model: ColabDesign AF model instance containing trajectory data
        design_name (str): Unique identifier for this design trajectory
        io (IO): IO handler instance for file path management
        
    Note:
        Currently focuses on overall loss and IgLM likelihood metrics. Additional
        metrics can be enabled by modifying the metrics_to_plot list.
    """
    # metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'helix', 'beta_strand3', 'iglm_ll']
    metrics_to_plot = ["loss", "iglm_ll"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    plots_dir = os.path.join(io.layout.trajectories, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            # Create a new figure for each metric
            plt.figure()

            loss = af_model.get_loss(metric)
            # Create an x axis for iterations
            iterations = range(1, len(loss) + 1)

            plt.plot(
                iterations, loss, label=f"{metric}", color=colors[index % len(colors)]
            )

            # Add labels and a legend
            plt.xlabel("Iterations")
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)

            # Save the plot
            plt.savefig(
                os.path.join(
                    plots_dir, design_name + "_" + metric + ".png"
                ),
                dpi=150,
            )
            # save loss values if iglm ll
            if metric == "iglm_ll":
                with open(
                    os.path.join(
                        plots_dir, design_name + "_iglm_ll.txt"
                    ),
                    "w",
                ) as f:
                    for item in loss:
                        f.write("%s\n" % item)

            # Close the figure
            plt.close()


