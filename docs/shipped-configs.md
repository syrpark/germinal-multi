# Shipped configurations

Three campaigns, each a two-state co-optimisation, each provided for both
design styles. The configs vary along three axes: **which target**, **which
design style**, and **whether it is a rescue run**.

| Campaign | Design style | Run config | Target config | Initial filters |
|---|---|---|---|---|
| EGFR domain III (WT + mutant) | Germinal | `multi` | `egfr` | `scfv_multi` |
| EGFR domain III (WT + mutant) | Germinal 1.1 | `multi_logits` | `egfr` | `scfv_multi` |
| SARS-CoV-2 RBD (BA.2 + KP.3) | Germinal | `multi` | `ba2_kp3` | `scfv_multi` |
| SARS-CoV-2 RBD (BA.2 + KP.3) | Germinal 1.1 | `multi_logits` | `ba2_kp3` | `scfv_multi` |
| XFG RBD (WT + 3-mut challenge) | Germinal | `multi_rescue` | `xfg` | `scfv_multi_rescue` |
| XFG RBD (WT + 3-mut challenge) | Germinal 1.1 | `multi_rescue_logits` | `xfg` | `scfv_multi_rescue` |

All six use `filter/final=scfv_multi`. The hallucination gates are identical in
every run config — `plddt_threshold: 0.80`, `i_ptm_threshold: 0.73`,
`i_pae_threshold: 0.22` — so no per-campaign overrides are needed.

## Axis 1 — the target

The target config supplies one entry per antigen state: the structures, the
hotspot sets and the names. Nothing else changes with the target.

| Target config | Antigen states |
|---|---|
| `egfr` | `egfr`, `egfr_mut` |
| `ba2_kp3` | `ba2`, `kp3` |
| `xfg` | `xfg_rbd_wt`, `xfg_rbd_3mut` |

## Axis 2 — the design style

| | Germinal (`multi`, `multi_rescue`) | Germinal 1.1 (`multi_logits`, `multi_rescue_logits`) |
|---|---|---|
| `logits_only` | `false` | `true` |
| `recycling` | `false` | `true` |
| Hallucination | logits → softmax → semi-greedy | logits only |
| `logits_steps` | 40 (`multi`) / 45 (`multi_rescue`) | 45 |
| `softmax_steps` | 10 | 0 |
| `search_steps` | 5 | 0 |
| `num_recycles` | — (unused) | 5 |
| `max_mpnn_sequences` | 4 | 1 |
| `interface_ratchet` | — | `true` |

The two differ in where the search effort goes. Germinal spends it on the
softmax and semi-greedy stages after the logits phase; Germinal 1.1 stops after
logits and spends it instead on the redesign↔refold loop, taking one AbMPNN
sequence per cycle over 5 cycles rather than a single fan-out of 4.

## Axis 3 — rescue

The two `multi_rescue*` configs redesign an existing antibody rather than
designing one from scratch. Four things change:

1. **The whole starting complex is given, not just the antigen.**
   `complex_path` points at the bound bd55_1205 scFv–antigen structures, one
   per antigen state:
   ```yaml
   complex_path: ["pdbs/bd55_xfg.pdb",
                  "pdbs/bd55_xfg_next.pdb"]
   ```
2. **Each trajectory starts from bd55_1205** rather than a random sequence:
   `is_recover: true`, with `seq_init_mode: []` so nothing overrides that seed.
   The binder geometry follows the scaffold, so `cdr_lengths`, `fw_lengths` and
   `vh_len` describe a 241-residue binder, not the generic 242.
3. **The interface template is kept.** `rm_template_ic: false` preserves the
   inter-chain contacts (and `rm_binder_seq: false` the binder sequence), so the
   WT binding mode from `complex_path` is available to the model. The
   non-rescue configs strip both.
4. **The CDRs carry a logit bias toward the starting sequence.**
   `recover_bias: 5` is applied at exactly the designable positions: on the
   non-rescue path those rows of the bias are zeroed, while under `is_recover`
   they are set to `recover_bias` times the wild-type one-hot. The effect is to
   hold each CDR near bd55_1205 unless the loss pays to move it.

## Running them

```bash
# EGFR, Germinal
python -u run_germinal_multi.py run=multi target=egfr \
    filter/initial=scfv_multi filter/final=scfv_multi \
    experiment_name=egfr_germinal

# EGFR, Germinal 1.1
python -u run_germinal_multi.py run=multi_logits target=egfr \
    filter/initial=scfv_multi filter/final=scfv_multi \
    experiment_name=egfr_logits

# XFG antibody rescue, Germinal 1.1
python -u run_germinal_multi.py run=multi_rescue_logits target=xfg \
    filter/initial=scfv_multi_rescue filter/final=scfv_multi \
    experiment_name=xfg_rescue_logits

# XFG antibody rescue, original Germinal
python -u run_germinal_multi.py run=multi_rescue target=xfg \
    filter/initial=scfv_multi_rescue filter/final=scfv_multi \
    experiment_name=xfg_rescue_germinal
```

Swap `target=ba2_kp3` for the SARS-CoV-2 campaign. On the default backends the
AF3 install paths must be set first — see [AlphaFold3 setup](af3-setup.md).

## Filtering

`filter/initial` applies on both design styles. Every run does the cheap
built-in early check (CA clashes, hotspot proximity, CDR fraction, driven by
`run` keys) and then the `filter/initial` set on the cofolded structure. The
early check is a free pre-screen for the same three properties the initial
filter measures properly, so it mostly saves cofolding cost on designs that
would fail anyway.

`scfv_multi_rescue` is `scfv_multi` plus one extra gate, `initial.plddt >= 0.8`,
which applies to the two XFG rescue configs and nowhere else.

## Before you change a pairing

A run config and a target config are **not** freely interchangeable: the run
config carries the binder geometry (`cdr_lengths`, `fw_lengths`, `vh_len`, and
for rescue the `complex_path`), and that must match the binder in the starting
complex. `config.validate_binder_length` checks the lengths at startup --
`sum(cdr_lengths) + sum(fw_lengths)` against the binder chain -- and fails
loudly, rather than letting every CDR window silently shift. It does **not**
check `vh_len`, which is consumed separately by the IgLM prep.

- `multi` and `multi_logits` are the generic 242-residue scFv templates, and
  pair with `egfr` or `ba2_kp3`.
- The `multi_rescue*` configs must be paired with `xfg`: they carry the
  bd55_1205 scaffold geometry and the structures that supply it live in the
  target config. Pairing either with a generic `multi*` counterpart is a length
  mismatch and will be rejected at startup.
