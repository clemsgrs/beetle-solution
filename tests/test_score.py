import json

import pytest

from beetle import score


def _splits(tmp_path):
    path = tmp_path / "roi_splits.csv"
    rows = ["sample_id,split,fold"]
    for fold in range(5):
        rows += [
            f"roi-{fold},test,{fold}",
            f"roi-{(fold + 1) % 5},tune,{fold}",
            f"roi-{(fold + 2) % 5},train,{fold}",
        ]
    path.write_text("\n".join(rows) + "\n")
    return path


def _run_dir(tmp_path):
    run = tmp_path / "run"
    for fold in range(5):
        (run / f"fold_{fold}").mkdir(parents=True)
        (run / f"fold_{fold}/best_model.pt").write_bytes(f"checkpoint-{fold}".encode())
    return run


def _context(manifest, features):
    return {f"roi-{fold}": f"record-{fold}" for fold in range(5)}, "store"


def test_split_ids_follow_each_models_held_out_fold(tmp_path):
    splits = _splits(tmp_path)

    assert score.read_split_sample_ids(splits, split="test") == {
        fold: [f"roi-{fold}"] for fold in range(5)
    }
    assert score.read_split_sample_ids(splits, split="tune")[0] == ["roi-1"]


def test_scoring_resumes_by_skipping_folds_with_metrics(tmp_path):
    output = tmp_path / "scores"
    done = output / "attempt-01/fold_0/metrics_test.json"
    done.parent.mkdir(parents=True)
    done.write_text("{}")
    calls = []

    def scorer(**kwargs):
        calls.append((kwargs["fold"], kwargs["records"], kwargs["split"]))
        return {"dataset_global_mean_dice": 0.9}

    summary = score.score_attempts(
        attempts=[("attempt-01", _run_dir(tmp_path))],
        roi_manifest=tmp_path / "roi_manifest.csv",
        roi_splits=_splits(tmp_path),
        feature_dir=tmp_path,
        output_dir=output,
        context_builder=_context,
        scorer=scorer,
    )

    assert calls == [(fold, [f"record-{fold}"], "test") for fold in (1, 2, 3, 4)]
    assert summary["skipped"] == [{"attempt": "attempt-01", "fold": 0}]
    assert summary["scored"] == [{"attempt": "attempt-01", "fold": f} for f in (1, 2, 3, 4)]
    written = json.loads((output / "attempt-01/fold_4/metrics_test.json").read_text())
    assert written["roi_count"] == 1
    assert written["metrics"] == {"dataset_global_mean_dice": 0.9}
    assert written["checkpoint"]["bytes"] == len(b"checkpoint-4")


def test_scoring_refuses_rois_missing_from_the_manifest(tmp_path):
    output = tmp_path / "scores"

    with pytest.raises(ValueError, match="absent from the ROI manifest"):
        score.score_attempts(
            attempts=[("attempt-01", _run_dir(tmp_path))],
            roi_manifest=tmp_path / "roi_manifest.csv",
            roi_splits=_splits(tmp_path),
            feature_dir=tmp_path,
            output_dir=output,
            context_builder=lambda manifest, features: ({}, "store"),
            scorer=lambda **kwargs: pytest.fail("scored without manifest records"),
        )
    assert not list(output.rglob("metrics_test.json"))


def test_finished_attempts_do_not_load_the_feature_cache(tmp_path):
    output = tmp_path / "scores"
    for fold in range(5):
        done = output / f"attempt-01/fold_{fold}/metrics_test.json"
        done.parent.mkdir(parents=True)
        done.write_text("{}")

    summary = score.score_attempts(
        attempts=[("attempt-01", _run_dir(tmp_path))],
        roi_manifest=tmp_path / "roi_manifest.csv",
        roi_splits=_splits(tmp_path),
        feature_dir=tmp_path,
        output_dir=output,
        context_builder=lambda manifest, features: pytest.fail("loaded the feature cache"),
    )

    assert summary["scored"] == []
    assert len(summary["skipped"]) == 5
