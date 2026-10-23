import json

import numpy as np
import pytest

from morphing_rovers.chromosome import load_chromosome, save_chromosome
from morphing_rovers.cli import main
from morphing_rovers.submission import CHALLENGE_ID, PROBLEM_ID, write_submission


def tiny_run_overrides(tmp_path, artifacts) -> list[str]:
    overrides = [f"output_dir={tmp_path}", f"masks.autoencoder.checkpoint={artifacts / 'autoencoder.pt'}",
                 "masks.dataset.samples_per_scenario=2", "masks.optimization.n_iter=5",
                 "masks.optimization.refinement_rounds=1", "imitation.scenarios=[0]", "imitation.n_runs=1",
                 "imitation.horizon_stop=2", "imitation.n_iter=2"]
    return [arg for override in overrides for arg in ("-s", override)]


@pytest.mark.slow
def test_run(tmp_path, artifacts, capsys):
    main(["run", *tiny_run_overrides(tmp_path, artifacts)])
    assert "Avg. fitness" in capsys.readouterr().out
    for name in ("config.yml", "masks.npy", "history.csv", "best_chromosome.npy", "submission.json"):
        assert (tmp_path / name).exists()
    masks = np.load(tmp_path / "masks.npy")
    np.testing.assert_array_equal(load_chromosome(tmp_path / "best_chromosome.npy")[:484], masks.ravel())
    with open(tmp_path / "submission.json") as f:
        assert json.load(f)[0]["decisionVector"] == load_chromosome(tmp_path / "best_chromosome.npy").tolist()


def test_sweep(tmp_path, artifacts):
    main(["sweep", *tiny_run_overrides(tmp_path, artifacts), "-s", "sweep.scenarios=[19]",
          "-s", "sweep.radius_factors=[null, 2.0]"])
    with open(tmp_path / "sweep.json") as f:
        results = json.load(f)
    assert list(results) == ["19"] and list(results["19"]) == ["straight", "2.0"]


def test_evaluate(artifacts, capsys):
    main(["evaluate", str(artifacts / "chromosomes" / "chromosome_2.0211.npy"), "--scenarios", "0"])
    out = capsys.readouterr().out
    assert "2.088" in out and "Avg. fitness: 2.088" in out


def test_submit(tmp_path, solution):
    main(["submit", "--name", "test", "-o", str(tmp_path / "s.json"), str(_saved(tmp_path, solution))])
    with open(tmp_path / "s.json") as f:
        (submission,) = json.load(f)
    assert submission["challenge"] == CHALLENGE_ID and submission["problem"] == PROBLEM_ID
    assert submission["name"] == "test" and submission["decisionVector"] == solution.tolist()


def test_submit_rejects_or_clips_out_of_bounds(tmp_path, solution):
    invalid = solution.copy()
    invalid[628] = 10000
    with pytest.raises(ValueError, match="out of bounds"):
        write_submission(invalid, tmp_path / "s.json")
    main(["submit", "--clip", "-o", str(tmp_path / "s.json"), str(_saved(tmp_path, invalid))])
    with open(tmp_path / "s.json") as f:
        assert json.load(f)[0]["decisionVector"][628] == 100


def _saved(tmp_path, chromosome):
    save_chromosome(tmp_path / "x.npy", chromosome)
    return tmp_path / "x.npy"
