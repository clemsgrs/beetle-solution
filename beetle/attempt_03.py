"""Attempt 03 cache audit and numerical/GPU preflight; does not launch training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/attempts/attempt-02-cache-lock.json"
ATTEMPT = ROOT / "configs/attempts/attempt-03.yaml"
BATCHES = ((64, 1), (32, 2), (16, 4), (8, 8))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_cache(data_dir: Path, output: Path) -> dict:
    """Rehash every tensor and sidecar against the immutable Attempt 01 manifest."""
    output.unlink(missing_ok=True)
    lock = json.loads(LOCK.read_text())
    for filename, key in (("dataset.csv", "dataset_sha256"),
                          ("splits.csv", "splits_sha256")):
        if sha256(data_dir / "curated_slide_manifest" / filename) != lock[key]:
            raise ValueError(f"Locked {filename} identity mismatch")
    manifest = data_dir / lock["manifest_file"]
    if sha256(manifest) != lock["manifest_sha256"]:
        raise ValueError("Locked cache manifest identity mismatch")
    feature_dir = data_dir / "cache" / lock["cache_root_name"] / lock["feature_namespace"]
    entries = []
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Cache manifest path escapes feature directory")
        entries.append((expected, relative))
    expected_paths = {name for _, name in entries}
    observed_paths = {p.relative_to(feature_dir) for p in feature_dir.rglob("*")
                      if p.is_file() and (p.suffix == ".pt" or p.name.endswith(".meta.json"))}
    if len(entries) != lock["manifest_entries"] or len(expected_paths) != len(entries):
        raise ValueError("Cache manifest count or uniqueness mismatch")
    if observed_paths != expected_paths:
        raise ValueError("Cache payload coverage differs from locked manifest")

    def check(entry):
        expected, relative = entry
        path = feature_dir / relative
        if sha256(path) != expected:
            raise ValueError(f"Cache checksum mismatch: {relative}")
        return path.suffix == ".pt", path.stat().st_size

    started = perf_counter()
    tensors = sidecars = tensor_bytes = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, (is_tensor, size) in enumerate(executor.map(check, entries), 1):
            tensors += int(is_tensor)
            sidecars += int(not is_tensor)
            tensor_bytes += size if is_tensor else 0
            if index % 10000 == 0:
                print(f"cache checksums: {index}/{len(entries)}", flush=True)
    if (tensors, sidecars, tensor_bytes) != (
        lock["tensor_files"], lock["sidecar_files"], lock["payload_bytes"]
    ):
        raise ValueError("Cache payload counts or bytes differ from lock")
    result = dict(status="completed", attempt_id="attempt-03",
                  method="sha256_every_tensor_and_sidecar", feature_dir=str(feature_dir.resolve()),
                  manifest_sha256=lock["manifest_sha256"], dataset_sha256=lock["dataset_sha256"],
                  splits_sha256=lock["splits_sha256"], tensor_files=tensors,
                  sidecar_files=sidecars, payload_bytes=tensor_bytes,
                  elapsed_seconds=perf_counter() - started)
    write_json(output, result)
    return result


def validate_protocol():
    from beetle.attempts import load_attempt_config

    baseline = load_attempt_config(ROOT / "configs/attempts/attempt-01.yaml")
    candidate = load_attempt_config(ATTEMPT)
    expected = asdict(baseline)
    expected.update(output_root="data/beetle/runs/attempt-03",
                    tags=["beetle", "virchow2", "attempt-03"])
    expected["decoder"] = {"name": "heavy_conv", "params": {
        "hidden_dim": 256, "num_upsample_blocks": 2, "num_groups": 32,
        "pool_scales": [1, 2, 3, 6]}}
    if asdict(candidate) != expected:
        raise ValueError("Attempt 03 scientific protocol differs beyond the agreed decoder change")
    return candidate


def probe(physical: int, accumulation: int, device: str, feature_path: Path) -> dict:
    import torch
    from soma.decoders.registry import build_decoder_for_grid
    from soma.dense.geometry import compute_dense_geometry
    from soma.tasks.segmentation import SegmentationHead
    from soma.training.model import SegmentationModel

    config = validate_protocol()
    if (physical, accumulation) not in BATCHES:
        raise ValueError("Unsupported effective-batch candidate")
    torch.set_num_threads(4)
    torch.manual_seed(config.training.seed)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(config.training.seed)
    geometry = compute_dense_geometry(target_size=512, patch_size=14)

    def build():
        decoder = build_decoder_for_grid(config.decoder.name, config.decoder.params,
                                        geometry=geometry, input_dim=1280, num_classes=4)
        return SegmentationModel(decoder=decoder,
                                 task_head=SegmentationHead(num_classes=4, geometry=geometry)).to(device)

    model = build().train()
    cached = torch.load(feature_path, map_location="cpu", weights_only=True)
    if not isinstance(cached, torch.Tensor) or tuple(cached.shape) != (1280, 37, 37):
        raise ValueError(f"Unexpected cached tensor format: {type(cached)}")
    if cached.dtype != torch.float16 or not torch.isfinite(cached).all():
        raise ValueError("Expected finite FP16 cached features")
    features = cached.float().unsqueeze(0).repeat(physical, 1, 1, 1).to(device)
    # Synthetic targets deliberately exercise all four classes and ignored pixels.
    masks = torch.arange(512 * 512, device=device).remainder(4).reshape(1, 512, 512).repeat(physical, 1, 1)
    masks[:, -1, :] = 255
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate,
                                 weight_decay=config.training.weight_decay)
    before = next(model.parameters()).detach().clone()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = perf_counter()
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            logits = model(features).logits
            if tuple(logits.shape) != (physical, 4, 512, 512):
                raise ValueError("Decoder/head output geometry mismatch")
            loss = model.task_head.compute_loss(logits, {"mask": masks})
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            (loss / accumulation).backward()
            losses.append(loss.item())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Missing or non-finite gradients")
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    if torch.equal(before, next(model.parameters()).detach()):
        raise ValueError("Optimizer did not update parameters")
    model.eval()
    with torch.no_grad():
        reference = model(features[:1]).logits.clone()
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = build().eval()
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(features[:1]).logits, reference, rtol=0, atol=0)
    return dict(passed=True, physical_batch_size=physical, accumulation_steps=accumulation,
                effective_batch_size=physical * accumulation, optimizer_steps=2,
                parameters=sum(p.numel() for p in model.parameters()), losses=losses,
                finite_gradients=True, parameters_changed=True, checkpoint_reload_exact=True,
                logits_shape=list(logits.shape), feature_path=str(feature_path),
                targets="synthetic_four_class_with_ignore_255", training_dtype="float32",
                elapsed_seconds=elapsed, rois_per_second=128 / elapsed,
                peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                free_bytes_before_updates=free_bytes, total_memory_bytes=total_bytes,
                gpu=torch.cuda.get_device_name(), torch_version=torch.__version__)


def prepare(data_dir: Path, evidence_dir: Path, output_root: Path) -> Path:
    """Write a launch configuration only when cache and numerical preflight agree."""
    from beetle.attempts import load_attempt_config
    from soma.config import load_config, save_config

    current = validate_protocol()
    frozen = load_config(evidence_dir / "resolved-attempt-03.yaml")
    if asdict(current) != asdict(frozen):
        raise ValueError("Attempt 03 protocol changed after GPU preflight")
    cache = json.loads((evidence_dir / "cache-validation.json").read_text())
    preflight = json.loads((evidence_dir / "preflight.json").read_text())
    lock = json.loads(LOCK.read_text())
    feature_dir = data_dir / "cache" / lock["cache_root_name"] / lock["feature_namespace"]
    if (cache.get("status") != "completed" or cache.get("attempt_id") != "attempt-03"
            or preflight.get("status") != "completed" or preflight.get("attempt_id") != "attempt-03"
            or cache.get("feature_dir") != str(feature_dir.resolve())):
        raise ValueError("Missing or inconsistent pretraining evidence")
    for key in ("manifest_sha256", "dataset_sha256", "splits_sha256",
                "tensor_files", "sidecar_files", "payload_bytes"):
        if cache.get(key) != lock[key]:
            raise ValueError(f"Cache audit disagrees with locked {key}")
    for filename, key in (("dataset.csv", "dataset_sha256"), ("splits.csv", "splits_sha256")):
        if sha256(data_dir / "curated_slide_manifest" / filename) != lock[key]:
            raise ValueError(f"{filename} changed after cache audit")
    selected = preflight["execution"]["selected"]
    physical, accumulation = selected["physical_batch_size"], selected["accumulation_steps"]
    if (physical, accumulation) not in BATCHES or selected["effective_batch_size"] != 64:
        raise ValueError("Invalid frozen batch selection")
    passing = [r for r in preflight["execution"]["candidates"] if r.get("passed") is True]
    if not passing or any(passing[0][key] != selected[key] for key in selected):
        raise ValueError("Selected batch disagrees with passing probes")
    config = load_attempt_config(ATTEMPT, overrides={
        "data": {"dataset_csv": str((data_dir / "curated_slide_manifest/dataset.csv").resolve()),
                 "splits_csv": str((data_dir / "curated_slide_manifest/splits.csv").resolve())},
        "cache": {"root_dir": str((data_dir / "cache" / lock["cache_root_name"]).resolve())},
        "run": {"output_root": str(output_root.resolve()), "run_id": "attempt-03-heavy-conv"},
        "training": {"batch_size": physical, "gradient_accumulation": accumulation},
    })
    path = evidence_dir / "launch/attempt-03.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_config(config, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cache = sub.add_parser("validate-cache")
    cache.add_argument("--data-dir", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    worker = sub.add_parser("probe-worker")
    worker.add_argument("--physical", type=int, required=True)
    worker.add_argument("--accumulation", type=int, required=True)
    worker.add_argument("--device", default="cuda:0")
    worker.add_argument("--feature", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--feature", type=Path, required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    preflight.add_argument("--device", default="cuda:0")
    preparation = sub.add_parser("prepare")
    preparation.add_argument("--data-dir", type=Path, required=True)
    preparation.add_argument("--evidence-dir", type=Path, required=True)
    preparation.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate-cache":
        validate_cache(args.data_dir, args.output)
    elif args.command == "probe-worker":
        write_json(args.output, probe(args.physical, args.accumulation, args.device, args.feature))
    elif args.command == "prepare":
        print(prepare(args.data_dir, args.evidence_dir, args.output_root))
    else:
        from soma.config import save_config

        config = validate_protocol()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "preflight.json").unlink(missing_ok=True)
        results = []
        for physical, accumulation in BATCHES:
            print(f"probing {physical}x{accumulation}", flush=True)
            worker_output = args.output_dir / f"probe-{physical}x{accumulation}.json"
            worker_output.unlink(missing_ok=True)
            try:
                completed = subprocess.run([sys.executable, "-m", "beetle.attempt_03", "probe-worker",
                    "--physical", str(physical), "--accumulation", str(accumulation),
                    "--device", args.device, "--feature", str(args.feature),
                    "--output", str(worker_output)],
                    text=True, capture_output=True, timeout=900)
            except subprocess.TimeoutExpired:
                completed = subprocess.CompletedProcess([], 1, "", "Probe exceeded 900 seconds")
            if completed.returncode:
                result = dict(passed=False, physical_batch_size=physical,
                              accumulation_steps=accumulation, error=completed.stderr[-8000:])
            else:
                result = json.loads(worker_output.read_text())
            print(f"{physical}x{accumulation}: {'passed' if result['passed'] else 'failed'}", flush=True)
            results.append(result)
            write_json(args.output_dir / "gpu-probes.json", {"candidates": results})
        passing = [r for r in results if r["passed"]]
        if not passing:
            raise RuntimeError("No GPU batch candidate passed; see gpu-probes.json")
        selected = passing[0]
        save_config(config, args.output_dir / "resolved-attempt-03.yaml")
        write_json(args.output_dir / "preflight.json", dict(
            status="completed", attempt_id="attempt-03", scientific_protocol="decoder_only_change",
            cache_validation="separate cache-validation.json required before training",
            execution={"candidates": results, "selected": {k: selected[k] for k in
                       ("physical_batch_size", "accumulation_steps", "effective_batch_size")},
                       "frozen_fold_ids": [0, 1, 2, 3, 4]},
            roi_draws_per_epoch_by_fold=json.loads(LOCK.read_text())["roi_draws_per_epoch_by_fold"]))


if __name__ == "__main__":
    main()
