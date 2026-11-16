"""Pins candidate fold batching: every candidate folded once, no slot split.

The chunker is imported directly; the loop around it lives in main(), which
needs a GPU, so that part is asserted at the source level.
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from germinal.filters import folding
from germinal.utils import utils

PASSED = 0
FAILED = 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name} {detail}")
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise AssertionError(f"{name} {detail}")


def _chunker():
    return utils.chunks_by_candidate_count


def _slots(*counts):
    return [
        {"design_name": f"d{i}", "candidates": [{"name": f"d{i}_abmpnn_{j}"}
                                                for j in range(n)]}
        for i, n in enumerate(counts)
    ]


def _names(chunks):
    return [c["name"] for chunk in chunks for slot in chunk
            for c in slot["candidates"]]


def test_every_candidate_is_folded_exactly_once():
    chunk_by = _chunker()
    slots = _slots(5, 5, 5, 5, 5, 5, 5, 5, 5, 5)
    for cap in (0, 1, 3, 5, 10, 11, 50, 1000):
        chunks = chunk_by(slots, cap)
        got = _names(chunks)
        check(
            f"cap={cap} folds all 50 candidates once",
            sorted(got) == sorted(_names([slots])) and len(got) == 50,
            f"{len(got)} candidates",
        )


def test_no_chunk_exceeds_the_cap_unless_one_slot_does():
    chunk_by = _chunker()
    slots = _slots(3, 3, 3, 3, 3, 3)
    for cap in (3, 6, 9):
        for chunk in chunk_by(slots, cap):
            n = sum(len(s["candidates"]) for s in chunk)
            check(f"cap={cap}: a chunk holds {n} <= {cap}", n <= cap, n)


def test_no_chunk_is_empty():
    chunk_by = _chunker()
    for cap in (0, 1, 4, 7):
        chunks = chunk_by(_slots(2, 2, 2, 2), cap)
        check(f"cap={cap}: no empty chunk", all(chunks), chunks)


def test_a_slot_is_never_split():
    # Its candidates share one any_accepted and one elapsed total.
    chunk_by = _chunker()
    slots = _slots(5, 5)
    for cap in (1, 2, 3, 4, 5, 6):
        chunks = chunk_by(slots, cap)
        for chunk in chunks:
            for slot in chunk:
                check(
                    f"cap={cap}: {slot['design_name']} stays whole",
                    len(slot["candidates"]) == 5,
                )
        check(f"cap={cap}: both slots appear once",
              sorted(s["design_name"] for c in chunks for s in c)
              == ["d0", "d1"])


def test_an_oversized_slot_gets_its_own_chunk():
    chunk_by = _chunker()
    chunks = chunk_by(_slots(2, 20, 2), 5)
    big = [c for c in chunks if any(len(s["candidates"]) == 20 for s in c)]
    check("the 20-candidate slot is alone in its chunk",
          len(big) == 1 and len(big[0]) == 1, chunks)


def test_a_zero_or_absent_cap_means_one_chunk():
    chunk_by = _chunker()
    for cap in (0, None):
        chunks = chunk_by(_slots(4, 4, 4), cap)
        check(f"cap={cap!r} folds everything in one call", len(chunks) == 1,
              len(chunks))


def test_no_slots_means_no_folds():
    chunk_by = _chunker()
    check("an empty batch yields no chunks", chunk_by([], 10) == [])
    check("and no chunks when uncapped either", chunk_by([], 0) == [])


def test_slots_with_no_candidates_still_round_trip():
    chunk_by = _chunker()
    chunks = chunk_by(_slots(0, 0), 5)
    check("empty-candidate slots are not dropped",
          sorted(s["design_name"] for c in chunks for s in c) == ["d0", "d1"],
          chunks)


def _folder(backend="af3", **over):
    """One instance of the configs/folding group, as Hydra would compose it."""
    f = {
        "backend": backend,
        "num_samples": 5,
        "num_cycles": 10,
        "sample_select_mode": "best",
        "seed_select_mode": "worst",
        "msa_method": "colabfold_remote",
        "msa_scope": "target_only",
        "use_msa": True,
        "batch_size": 10,
    }
    f.update(over)
    return f


def _settings(**over):
    s = {"filter_folder": _folder(), "recycle_folder": _folder()}
    s.update(over)
    return s


def test_the_batch_size_is_read_per_stage():
    s = _settings(filter_folder=_folder(batch_size=4),
                  recycle_folder=_folder(batch_size=25))
    check("the filter stage reads its own key",
          folding.resolve_batch_size(s, "filter") == 4)
    check("and the recycle stage reads its own",
          folding.resolve_batch_size(s, "recycle") == 25)


def test_an_absent_batch_size_defaults():
    folder = _folder()
    del folder["batch_size"]
    check("a config without the key still resolves",
          folding.resolve_batch_size(_settings(filter_folder=folder), "filter")
          == folding.DEFAULT_BATCH_SIZE)


def test_zero_means_one_call():
    s = _settings(filter_folder=_folder(batch_size=0))
    check("zero resolves to zero", folding.resolve_batch_size(s, "filter") == 0)
    check("and the chunker folds everything at once",
          len(_chunker()(_slots(4, 4, 4), 0)) == 1)


def test_a_nonsense_batch_size_raises():
    for bad in ("ten", None, -1, [4]):
        s = _settings(filter_folder=_folder(batch_size=bad))
        try:
            folding.resolve_batch_size(s, "filter")
        except ValueError:
            check(f"batch_size={bad!r} is rejected", True)
        except TypeError:
            check(f"batch_size={bad!r} raised TypeError, wanted ValueError", False)
        else:
            check(f"batch_size={bad!r} was accepted", False)


def test_the_shipped_folding_configs_declare_it():
    import yaml

    for name in ("af3", "protenix"):
        cfg = yaml.safe_load((REPO / "configs" / "folding" / f"{name}.yaml").read_text())
        check(f"{name}.yaml declares batch_size", "batch_size" in cfg, sorted(cfg))
        check(f"{name}.yaml's batch_size is a positive int",
              isinstance(cfg.get("batch_size"), int) and cfg["batch_size"] > 0)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
