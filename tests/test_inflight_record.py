"""Pins the in-flight journal: what a record holds and what it must not claim.

A record says "this design passed hallucination and has not finished". It is
deliberately not a row in all_trajectories.csv, and the tests here hold that
line: a record must never make check_existing_seed report a seed as spent.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from germinal.utils.io import (
    INFLIGHT_SCHEMA,
    IO,
    RunLayout,
    Trajectory,
    config_fingerprints,
)

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


def _io(tmp):
    return IO(RunLayout.create(Path(tmp) / "run"))


def _record(phase="hallucinated", **over):
    rec = {
        "phase": phase,
        "seed": 488766,
        "sequence": "QVQLVE",
        "hallucination_pdbs": ["/t/egfr_scfv_s488766_target0.pdb"],
        "target_fp": "sha256:aaa",
        "filter_fp": "sha256:bbb",
        "candidates": [],
    }
    rec.update(over)
    return rec


def _settings():
    run = {
        "type": "scfv",
        "cdr_lengths": [8, 8, 13, 6, 3, 9],
        "fw_lengths": [25, 17, 38, 11],
        "logits_only": False,
        "recycling": True,
        "num_recycles": 3,
        "recycle_ipsae_threshold": 0.4,
        "early_check_min_cdr_fraction": 0.4,
        "filter_folder": {"backend": "af3"},
        "recycle_folder": {"backend": "af3"},
        "max_trajectories": 10000,
        "max_hallucination_passing": 10,
        "max_hallucination_attempts": 100,
    }
    targets = {
        "target_name": ["wt", "mut"],
        "target_pdb_path": ["/wt.pdb", "/mut.pdb"],
        "target_chain": "A",
        "binder_chain": "B",
        "target_hotspots": ["A70,A74", "A70,A74"],
        "num_targets": 2,
    }
    return run, targets, {"initial.plddt": {"operator": ">=", "value": 0.8}}, {
        "final.iptm": {"operator": ">=", "value": 0.73}
    }


def test_a_record_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.record_inflight("egfr_scfv_s488766", _record())
        got = io.read_inflight()
        check("one record comes back", len(got) == 1, got)
        check("design_name is stamped", got[0]["design_name"] == "egfr_scfv_s488766")
        check("schema is stamped", got[0]["schema"] == INFLIGHT_SCHEMA)
        check("the sequence survives", got[0]["sequence"] == "QVQLVE")
        check("pid and host are stamped", "pid" in got[0] and "host" in got[0])


def test_numpy_and_paths_survive_the_write():
    # The metric dicts hold np.float64 and PosixPath; json.dump raises on both.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        traj = Trajectory("d_s1", "exp", "8,8", "A70")
        traj.update_filtering_metrics({
            "initial.plddt": np.float64(0.87),
            "initial.n": np.int64(3),
            "initial.arr": np.array([1.0, 2.0]),
        })
        traj.set_final_struct(Path("/t/d_s1.pdb"))
        io.record_inflight("d_s1", {"phase": "filtered",
                                    "trajectory": traj.snapshot()})
        got = io.read_inflight()[0]["trajectory"]
        check("numpy float becomes a float",
              got["filtering_metrics"]["initial.plddt"] == 0.87)
        check("numpy int becomes an int",
              got["filtering_metrics"]["initial.n"] == 3)
        check("numpy array becomes a list",
              got["filtering_metrics"]["initial.arr"] == [1.0, 2.0])
        check("a Path becomes a string",
              got["final_struct"] == "/t/d_s1.pdb")


def test_a_snapshot_rebuilds_the_trajectory():
    traj = Trajectory("d_s1", "exp", "8,8,13", "A70,A74")
    traj.update_filtering_metrics({"initial.plddt": 0.87})
    traj.update_other_metrics({"status": "x", "num_recycles_completed": 4})
    traj.set_save_location("trajectories")
    traj.set_final_struct("/t/d.pdb")
    back = Trajectory.from_snapshot(traj.snapshot())
    check("design_name survives", back.design_name == "d_s1")
    check("cdr_lengths survive", back.cdr_lengths == "8,8,13")
    check("hotspots survive", back.target_hotspots == "A70,A74")
    check("filtering metrics survive",
          back.filtering_metrics == {"initial.plddt": 0.87})
    check("other metrics survive",
          back.other_metrics["num_recycles_completed"] == 4)
    check("save_location survives", back.save_location == "trajectories")
    check("final_struct survives", back.final_struct == "/t/d.pdb")
    check("the rebuilt row matches",
          back.get_trajectory() == traj.get_trajectory())


def test_a_later_phase_supersedes_rather_than_duplicates():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.record_inflight("d_s1", _record("hallucinated"))
        io.record_inflight("d_s1", _record(
            "candidates_ready", candidates=[{"name": "d_s1_abmpnn_1", "seq": "QV"}]
        ))
        got = io.read_inflight()
        check("still one record", len(got) == 1, got)
        check("it reports the later phase", got[0]["phase"] == "candidates_ready")
        check("it carries the candidates", len(got[0]["candidates"]) == 1)


def test_a_write_leaves_no_temp_file():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.record_inflight("d_s1", _record())
        leftovers = [p.name for p in Path(io.layout.inflight).iterdir()
                     if ".tmp." in p.name]
        check("no .tmp. sibling survives", leftovers == [], leftovers)


def test_clearing_removes_the_record_and_its_lock():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.record_inflight("d_s1", _record())
        io.clear_inflight("d_s1")
        check("the record is gone", io.read_inflight() == [])
        left = sorted(p.name for p in Path(io.layout.inflight).iterdir())
        check("and so is its lock file", left == [], left)


def test_clearing_a_missing_record_is_a_no_op():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.clear_inflight("never_existed")
        check("clearing nothing does not raise", True)


def test_an_unreadable_record_is_skipped_not_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.record_inflight("good_s1", _record())
        (Path(io.layout.inflight) / "truncated_s2.json").write_text('{"phase": ')
        got = io.read_inflight()
        check("the good record still loads", [r["design_name"] for r in got]
              == ["good_s1"], got)


def test_a_foreign_schema_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        rec = _record()
        rec["schema"] = 99
        (Path(io.layout.inflight) / "future_s1.json").write_text(json.dumps(rec))
        check("a record from another build is not read", io.read_inflight() == [])


def test_a_record_is_not_a_json_object():
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        (Path(io.layout.inflight) / "list_s1.json").write_text("[1, 2, 3]")
        check("a non-object record is skipped", io.read_inflight() == [])


def test_a_record_does_not_retire_the_seed():
    # The whole point of not writing in-progress rows: a crash must leave the
    # seed available, not mark it done with no result.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        for phase in ("hallucinated", "filtered", "candidates_ready"):
            io.record_inflight("egfr_scfv_s488766", _record(phase))
            check(
                f"a {phase} record does NOT mark the seed done",
                io.check_existing_seed(488766) is False,
            )


def test_fingerprints_split_target_from_filter():
    run, targets, initial, final = _settings()
    base = config_fingerprints(run, targets, initial)

    moved = dict(targets, target_hotspots=["A70,A99", "A70,A74"])
    check("a hotspot change moves target_fp",
          config_fingerprints(run, moved, initial)[0] != base[0])
    check("and leaves filter_fp alone",
          config_fingerprints(run, moved, initial)[1] == base[1])

    # An INITIAL threshold: that verdict is inherited by a candidates_ready
    # record, so it has to move the digest. The final filters are not
    # fingerprinted at all -- Phase D applies them to every adopted design,
    # so a changed one invalidates nothing and must not strand work.
    stricter = {"initial.iptm": {"operator": ">=", "value": 0.80}}
    check("an initial threshold change moves filter_fp",
          config_fingerprints(run, targets, stricter)[1] != base[1])
    check("and leaves target_fp alone",
          config_fingerprints(run, targets, stricter)[0] == base[0])

    backend = dict(run, filter_folder={"backend": "protenix"})
    check("a backend change moves filter_fp",
          config_fingerprints(backend, targets, initial)[1] != base[1])


def test_budget_knobs_are_not_fingerprinted():
    run, targets, initial, final = _settings()
    base = config_fingerprints(run, targets, initial)
    for key, value in (
        ("max_trajectories", 50),
        ("max_hallucination_passing", 4),
        ("max_hallucination_attempts", 10),
    ):
        changed = config_fingerprints(dict(run, **{key: value}),
                                      targets, initial)
        check(f"changing {key} invalidates nothing", changed == base)


def test_fingerprints_are_stable_across_calls():
    run, targets, initial, final = _settings()
    check("the same inputs give the same digests",
          config_fingerprints(run, targets, initial)
          == config_fingerprints(run, targets, initial))


def test_hallucination_metrics_reach_the_row_with_their_prefix():
    # They were silently dropped for every run in this repo's history: the
    # runner prefixes them, and the whitelist was matched against bare names.
    traj = Trajectory("d_s1", "exp", "8,8", "A70")
    traj.update_trajectory_metrics({
        "hallucination.plddt": 0.87,
        "hallucination.i_ptm": 0.44,
        "hallucination.i_pae": 0.31,
        "hallucination.loss": 2.12,
        "hallucination.iglm_ll": -2.79,
        "hallucination.rmsd": 34.6,
    })
    got = traj.trajectory_metrics
    check("the whitelisted metrics are kept",
          {"hallucination.plddt", "hallucination.i_ptm", "hallucination.i_pae",
           "hallucination.loss", "hallucination.iglm_ll"} <= set(got), sorted(got))
    check("the prefix survives, so AF3's initial./final. cannot be confused "
          "with the hallucinator's own numbers",
          all(k.startswith("hallucination.") for k in got), sorted(got))
    check("a metric outside the whitelist is still dropped",
          "hallucination.rmsd" not in got, sorted(got))
    check("and they reach the CSV row",
          traj.get_trajectory()["hallucination.plddt"] == 0.87)


def test_bare_metric_names_still_work():
    traj = Trajectory("d_s1", "exp", "8,8", "A70")
    traj.update_trajectory_metrics({"plddt": 0.9, "not_a_metric": 1})
    check("an unprefixed whitelisted metric is kept",
          traj.trajectory_metrics == {"plddt": 0.9}, traj.trajectory_metrics)

def test_recorded_structure_paths_are_absolute():
    # Stored relative, they resolve only from the directory the run started
    # in, so a second checkout -- or just an absolute results_dir -- made
    # adoption report "its structures are gone" for files still on disk.
    with tempfile.TemporaryDirectory() as tmp:
        io = _io(tmp)
        io.record_inflight("egfr_scfv_s1", _record(
            "candidates_ready",
            hallucination_pdbs=["results/x/a_target0.pdb"],
            final_struct=["results/x/b.pdb"],
            candidates=[{"name": "c1", "designed_on": ["results/x/c.pdb"]}],
        ))
        got = io.read_inflight()[0]
        check("hallucination_pdbs are absolute",
              all(os.path.isabs(p) for p in got["hallucination_pdbs"]),
              got["hallucination_pdbs"])
        check("final_struct is absolute",
              all(os.path.isabs(p) for p in got["final_struct"]),
              got["final_struct"])
        check("a candidate's designed_on is absolute too",
              all(os.path.isabs(p) for p in got["candidates"][0]["designed_on"]),
              got["candidates"][0]["designed_on"])
        check("and the basename is preserved, not rewritten",
              got["hallucination_pdbs"][0].endswith("results/x/a_target0.pdb"),
              got["hallucination_pdbs"][0])


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print(f"\n{PASSED}/{PASSED + FAILED} passed")
    sys.exit(1 if FAILED else 0)
