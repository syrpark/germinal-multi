"""Pins which unfinished designs a restarted run may take over.

Each test builds the on-disk state a crash would leave -- a record, some
structures, some rows -- and asserts the adoption decision. The windows that
matter are between hallucination and the first fold, inside redesign, and
part-way through scoring a candidate set.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.utils.io import IO, RunLayout

PASSED = 0
FAILED = 0

TARGET_FP = "sha256:target"
FILTER_FP = "sha256:filter"


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


def _io(tmp):
    return IO(RunLayout.create(Path(tmp) / "run"))


def _pdbs(io, design_name, n=2):
    """The hallucination output a crash leaves behind."""
    out = []
    for t in range(n):
        p = io.layout.trajectories / "structures" / f"{design_name}_target{t}.pdb"
        p.write_text("ATOM\n")
        out.append(str(p))
    return out


def _rows(io, *design_names):
    pd.DataFrame(
        [{"design_name": n, "status": "x"} for n in design_names]
    ).to_csv(io.layout.all_csv, index=False)


def _record(io, design_name, phase, candidates=(), pdbs=None, **over):
    rec = {
        "phase": phase,
        "seed": 488766,
        "sequence": "QVQLVE",
        "hallucination_pdbs": (
            _pdbs(io, design_name) if pdbs is None else pdbs
        ),
        "target_fp": TARGET_FP,
        "filter_fp": FILTER_FP,
        "candidates": [
            {"name": c, "seq": "QV", "score": 0.5, "seqid": 0.7,
             "source": "abmpnn", "designed_on": []}
            for c in candidates
        ],
    }
    rec.update(over)
    io.record_inflight(design_name, rec)
    return rec


def _adopt(io):
    return io.find_adoptable(TARGET_FP, FILTER_FP)


def _names(records):
    return [r["design_name"] for r in records]


# --- the crash windows ---

def test_a_hallucinated_design_with_no_row_is_adoptable():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "hallucinated")
        check("a design killed before its first fold is adoptable",
              _names(_adopt(io)) == ["d_s1"])


def test_a_design_killed_inside_redesign_is_adoptable():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "filtered")
        check("a filtered design with no row is adoptable",
              _names(_adopt(io)) == ["d_s1"])


def test_a_partly_scored_candidate_set_is_finished_not_dropped():
    # The quiet loss: candidate rows retire the parent's seed, so without
    # this the unscored candidates are gone for good.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "candidates_ready",
                candidates=[f"d_s1_abmpnn_{i}" for i in range(1, 6)])
        _rows(io, "d_s1_abmpnn_1", "d_s1_abmpnn_2")
        got = _adopt(io)
        check("the parent is adoptable", _names(got) == ["d_s1"], got)
        check(
            "and exactly the unscored candidates come back",
            [c["name"] for c in got[0]["missing_candidates"]]
            == ["d_s1_abmpnn_3", "d_s1_abmpnn_4", "d_s1_abmpnn_5"],
            got[0]["missing_candidates"],
        )


def test_a_fully_scored_candidate_set_is_not_adoptable():
    # A finished design never gets a row of its own, only candidate rows.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "candidates_ready",
                candidates=[f"d_s1_abmpnn_{i}" for i in range(1, 4)])
        _rows(io, "d_s1_abmpnn_1", "d_s1_abmpnn_2", "d_s1_abmpnn_3")
        check("a finished design is not mistaken for unfinished work",
              _adopt(io) == [])


def test_candidate_matching_is_anchored():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "candidates_ready",
                candidates=["d_s1_abmpnn_1", "d_s1_abmpnn_10"])
        _rows(io, "d_s1_abmpnn_10")
        got = _adopt(io)
        check("_abmpnn_10's row does not satisfy _abmpnn_1",
              [c["name"] for c in got[0]["missing_candidates"]]
              == ["d_s1_abmpnn_1"], got)


def test_a_recorded_design_is_not_adoptable():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "hallucinated")
        _rows(io, "d_s1")
        check("a design with its own row is done", _adopt(io) == [])


def test_candidate_rows_without_a_candidate_list_are_not_guessed_at():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "filtered")
        _rows(io, "d_s1_abmpnn_1")
        check("an ambiguous record is left alone", _adopt(io) == [])


# --- structures must still be there ---

def test_a_record_whose_structures_are_gone_is_not_adoptable():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        pdbs = _pdbs(io, "d_s1")
        _record(io, "d_s1", "hallucinated", pdbs=pdbs)
        os.remove(pdbs[1])
        check("one missing antigen state is disqualifying", _adopt(io) == [])


def test_a_record_with_no_structures_at_all_is_not_adoptable():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "hallucinated",
                pdbs=["/nonexistent/d_s1_target0.pdb"])
        check("a record pointing nowhere is skipped", _adopt(io) == [])


# --- config changes ---

def test_a_different_target_blocks_adoption_at_every_phase():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        for phase in ("hallucinated", "filtered", "candidates_ready"):
            _record(io, "d_s1", phase, candidates=["d_s1_abmpnn_1"],
                    target_fp="sha256:someone_else")
            check(f"{phase}: a different target is never adoptable",
                  _adopt(io) == [])


def test_a_different_filter_set_spares_hallucinated_work():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "hallucinated", filter_fp="sha256:other")
        check("nothing has been filtered yet, so it still adopts",
              _names(_adopt(io)) == ["d_s1"])


def test_a_different_filter_set_blocks_only_candidate_work():
    # Only candidates_ready inherits a verdict: it skips phases A-C, so its
    # initial-filter result and its recycle-built candidates come from the old
    # config. A filtered record re-enters at the fold and is re-filtered under
    # the current one, so stranding it threw away work that would have been
    # re-evaluated correctly anyway.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "filtered", filter_fp="sha256:other")
        check("a filtered record adopts despite a changed filter set, "
              "because adoption re-folds and re-filters it",
              _names(_adopt(io)) == ["d_s1"], _adopt(io))
        _record(io, "d_s2", "candidates_ready",
                candidates=["d_s2_abmpnn_1"], filter_fp="sha256:other")
        check("its candidates are still blocked, being the one verdict that "
              "would reach a row unrecomputed",
              "d_s2" not in _names(_adopt(io)), _adopt(io))


def test_a_skipped_record_is_kept_not_deleted():
    # Throwing away a human's unfinished work is not this function's call.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "filtered", target_fp="sha256:other")
        _adopt(io)
        check("the record survives being skipped",
              [r["design_name"] for r in io.read_inflight()] == ["d_s1"])


# --- ordering and the empty case ---

def test_adoption_is_oldest_first():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        # record_inflight stamps written_at itself, so write these by hand to
        # pin an order. Names are reversed to catch an alphabetical sort.
        for name, ts in (("d_s3", 300.0), ("d_s1", 100.0), ("d_s2", 200.0)):
            _record(io, name, "hallucinated")
            path = Path(io.layout.inflight) / f"{name}.json"
            rec = json.loads(path.read_text())
            rec["written_at"] = ts
            path.write_text(json.dumps(rec))
        check("the longest-waiting design goes first",
              _names(_adopt(io)) == ["d_s1", "d_s2", "d_s3"])


def test_a_clean_run_has_nothing_to_adopt():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        check("an empty inflight directory adopts nothing", _adopt(io) == [])


def test_adoption_survives_an_unreadable_master_csv():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "hallucinated")
        io.layout.all_csv.write_text("a,b\n1,2\n3,4,5\n")
        # Treating every design as unrecorded can only cost rework, which is
        # the right way to fail here.
        check("a corrupt CSV still yields a decision",
              _names(_adopt(io)) == ["d_s1"])


def test_a_candidate_whose_backbone_is_gone_is_dropped():
    # Scoring is against the backbone the candidate was designed on, so one
    # without it cannot be finished -- but its siblings still can.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        here = Path(tmp) / "bb.pdb"
        here.write_text("ATOM\n")
        _record(io, "d_s1", "candidates_ready")
        rec = json.loads((Path(io.layout.inflight) / "d_s1.json").read_text())
        rec["candidates"] = [
            {"name": "d_s1_abmpnn_1", "seq": "QV", "designed_on": [str(here)]},
            {"name": "d_s1_abmpnn_2", "seq": "QV", "designed_on": ["/gone.pdb"]},
        ]
        (Path(io.layout.inflight) / "d_s1.json").write_text(json.dumps(rec))
        got = _adopt(io)
        check("the scorable candidate survives",
              [c["name"] for c in got[0]["missing_candidates"]]
              == ["d_s1_abmpnn_1"], got)


def test_a_record_with_no_scorable_candidate_is_not_adopted():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "candidates_ready")
        rec = json.loads((Path(io.layout.inflight) / "d_s1.json").read_text())
        rec["candidates"] = [
            {"name": "d_s1_abmpnn_1", "seq": "QV", "designed_on": ["/gone.pdb"]},
        ]
        (Path(io.layout.inflight) / "d_s1.json").write_text(json.dumps(rec))
        check("nothing left to score means nothing to adopt", _adopt(io) == [])


def test_a_candidate_with_no_recorded_backbone_is_still_scorable():
    # The non-recycling path designs on the hallucination PDBs, already
    # checked above; an empty designed_on must not disqualify it.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        _record(io, "d_s1", "candidates_ready", candidates=["d_s1_abmpnn_1"])
        got = _adopt(io)
        check("an empty designed_on is not a missing backbone",
              [c["name"] for c in got[0]["missing_candidates"]]
              == ["d_s1_abmpnn_1"], got)


def test_a_changed_target_still_strands_every_phase():
    # The narrowing is to the filter gate only; target identity is unrelated.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        for phase in ("hallucinated", "filtered", "candidates_ready"):
            _record(io, f"d_{phase}", phase, candidates=["c_abmpnn_1"],
                    target_fp="sha256:a-different-antigen")
        check("no phase is adopted against a different target",
              _names(_adopt(io)) == [], _adopt(io))


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
