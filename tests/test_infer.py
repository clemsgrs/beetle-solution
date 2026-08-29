from types import SimpleNamespace

import soma.config
import soma.dense.live
import soma.dense.predict
import soma.pipeline

from beetle.infer import load_fold_predictor


def test_load_fold_predictor_uses_the_public_live_source_api(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text("unused by boundary fixture\n")
    for fold in range(5):
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir()
        (fold_dir / "best_model.pt").write_bytes(b"checkpoint")

    config = SimpleNamespace(
        decoder=SimpleNamespace(name="lightweight_conv", params={}),
        task=SimpleNamespace(params={"num_classes": 4}),
        encoder=SimpleNamespace(name="virchow2"),
        normalization=None,
        projection=None,
    )
    source = object()
    models = (object(),)
    predictor = object()

    monkeypatch.setattr(soma.config, "load_config", lambda path: config)
    monkeypatch.setattr(
        soma.dense.live, "build_live_segmentation_source", lambda observed: source
    )
    monkeypatch.setattr(
        soma.dense.predict,
        "build_live_segmentation_models",
        lambda *args, **kwargs: models,
    )

    class PredictorBoundary:
        @classmethod
        def from_source(cls, observed_source, observed_models):
            assert observed_source is source
            assert observed_models is models
            return predictor

    monkeypatch.setattr(
        soma.dense.predict, "SlidingWindowSegmentationPredictor", PredictorBoundary
    )

    def obsolete_pipeline(*args, **kwargs):
        raise AssertionError("inference must not use Soma's removed private Pipeline API")

    monkeypatch.setattr(soma.pipeline, "Pipeline", obsolete_pipeline)

    assert load_fold_predictor(run_dir) is predictor
