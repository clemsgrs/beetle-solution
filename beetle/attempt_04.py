"""Prepare and execute the frozen Mascaret encoder ablation in one CUDA process."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import gc
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import time

from beetle.attempts import REPO_ROOT, load_attempt_config
from beetle.gpu_job import claim_gpu, write_json

BASE = Path('/maindisk/clement/beetle-attempt-04-20260912')
REVISION = 'e95e7ea15e039e78d74def101415e19d9a67ba80'
DATA = Path('/maindisk/clement/soma-beetle-campaign-20260827/data/beetle')
ARCHIVE = Path('/maindisk/clement/beetle-attempt-03-20260908/evidence/replayed_rois')
# Soma run holding fold 0; relaunches must reuse it so completed folds are skipped.
RUN_ID = 'attempt-04-mascaret-20260912'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(base: Path) -> None:
    from soma.cache.roi_sampling import resolve_roi_sampling_cache, write_roi_sampling_coords
    from soma.config import save_config
    from soma.dataset import SegmentationManifest
    from soma.dense_slide_extraction import build_roi_manifest
    from soma.preprocessing.resolution import resolve_pipeline_preprocessing

    base.mkdir(parents=True, exist_ok=True)
    evidence = base / 'evidence'
    evidence.mkdir(exist_ok=True)
    config = load_attempt_config(REPO_ROOT / 'configs/attempts/attempt-04.yaml')
    lock = json.loads((REPO_ROOT / 'configs/attempts/attempt-02-cache-lock.json').read_text())
    for name, key in [('dataset.csv', 'dataset_sha256'), ('splits.csv', 'splits_sha256')]:
        if sha256(DATA / 'curated_slide_manifest' / name) != lock[key]:
            raise ValueError(f'Attempt 01 {name} identity mismatch')
    baseline = asdict(load_attempt_config(REPO_ROOT / 'configs/attempts/attempt-01.yaml'))
    candidate = asdict(config)
    # Verify every scientific setting except the explicit encoder/execution/path changes.
    for key in ('encoder', 'execution', 'cache', 'output_root', 'tags', 'dataset_csv', 'splits_csv'):
        baseline[key] = candidate[key]
    if baseline != candidate:
        raise ValueError('Unexpected scientific protocol change')
    if config.decoder.name != 'lightweight_conv' or config.decoder.params['num_upsample_blocks'] != 2:
        raise ValueError('Attempt 01 decoder topology must be retained')
    config = replace(config, run_id='attempt-04-mascaret-20260912')
    save_config(config, base / 'active-config.yaml')
    slides = SegmentationManifest(config.dataset_csv)
    coords = {sid: [] for sid in slides.sample_ids}
    stems = set()
    with (ARCHIVE / 'roi_manifest.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            sid, x, y = row['slide_id'], int(row['region_x']), int(row['region_y'])
            slide = slides.samples[sid]
            if (row['image_path'] != str(slide.image_path)
                    or row['label_mask_path'] != str(slide.label_mask_path)
                    or row['sample_id'] != f'{sid}__x{x}_y{y}'):
                raise ValueError('Archived ROI ancestry mismatch')
            coords[sid].append((x, y))
            stems.add(f'{Path(row["image_path"]).stem}/{x}_{y}.pt')
    payload_manifest = DATA / lock['manifest_file']
    if sha256(payload_manifest) != lock['manifest_sha256']:
        raise ValueError('Attempt 01 payload manifest identity mismatch')
    expected = {str(Path(line.split('  ', 1)[1])) for line in payload_manifest.read_text().splitlines()
                if line.endswith('.pt')}
    if stems != expected or sum(map(len, coords.values())) != 124697:
        raise ValueError('ROI population differs from Attempt 01')
    roi_manifest, roi_splits = build_roi_manifest(slides, config.splits_csv, coords,
                                                 out_dir=evidence / 'replayed_rois')
    hashes = {}
    for generated, name in [(roi_manifest, 'roi_manifest.csv'), (roi_splits, 'roi_splits.csv')]:
        hashes[name] = sha256(Path(generated))
        if hashes[name] != sha256(ARCHIVE / name):
            raise ValueError(f'Attempt 01 {name} replay differs')
    resolution = resolve_roi_sampling_cache(
        cache_root=Path(config.cache.root_dir), dataset=slides,
        preprocessing=resolve_pipeline_preprocessing(config))
    write_roi_sampling_coords(cache_resolution=resolution, coords_by_sample_id=coords)
    check = resolve_roi_sampling_cache(cache_root=Path(config.cache.root_dir), dataset=slides,
                                      preprocessing=resolve_pipeline_preprocessing(config))
    if not check.complete or check.coords_by_id != coords:
        raise ValueError('Replayed coordinate cache failed validation')
    weights = base / f'huggingface/hub/models--wearewaiv--mascaret/snapshots/{REVISION}/model.safetensors'
    runtime = {name: importlib.metadata.version(name) for name in
               ('soma-pathology', 'slide2vec', 'hs2p', 'torch', 'torchvision', 'numpy', 'Pillow', 'imagecodecs')}
    dist = importlib.metadata.distribution('soma-pathology')
    runtime['soma_source'] = json.loads(dist.read_text('direct_url.json'))
    write_json(evidence / 'preparation.json', {
        'status': 'completed', 'attempt_id': 'attempt-04', 'roi_count': 124697,
        'slides': len(coords), 'roi_hashes': hashes, 'runtime': runtime,
        'encoder_repository': 'wearewaiv/mascaret', 'encoder_revision': REVISION,
        'weight_sha256': sha256(weights), 'config_sha256': sha256(base / 'active-config.yaml'),
        'cache_root': str(config.cache.root_dir),
    })
    print('Preparation passed: exact Attempt 01 ROIs/splits, pinned Mascaret, fresh cache.', flush=True)


def smoke(base: Path, config) -> dict:
    import torch
    from slide2vec import Model, SlideRegions, DenseOptions
    from soma.slide2vec_adapter import build_execution_options
    from soma.decoders.registry import build_decoder_for_grid
    from soma.dense.geometry import compute_dense_geometry
    from soma.tasks.segmentation import SegmentationHead
    from soma.training.model import SegmentationModel

    torch.set_num_threads(4)
    torch.manual_seed(0)
    with (base / 'evidence/replayed_rois/roi_manifest.csv').open(newline='') as handle:
        row = next(csv.DictReader(handle))
    model = Model.from_preset('mascaret')
    execution = build_execution_options(
        config.encoder, execution=config.execution, encoder_name='mascaret',
        output_dir=base / 'smoke', num_gpus=1, save_tile_embeddings=True, output_dtype='fp16')
    started = time.monotonic()
    model.embed_regions_dense(
        [SlideRegions(sample_id=row['slide_id'], image_path=Path(row['image_path']),
                      coordinates=[(int(row['region_x']), int(row['region_y']))],
                      spacing_at_level_0=(float(row['spacing_at_level_0'])
                                          if row['spacing_at_level_0'] else None))],
        dense=DenseOptions(spacing_um=0.5, target_size=512, tolerance=0.1,
                           backend=config.preprocessing.backend, pad_mode='reflect',
                           window_size=224, overlap=0.5), execution=execution)
    paths = list((base / 'smoke/dense_embeddings').rglob('*.pt'))
    if len(paths) != 1:
        raise ValueError(f'Expected one smoke-test feature, got {len(paths)}')
    grid = torch.load(paths[0], weights_only=True, map_location='cpu')
    if tuple(grid.shape) != (1536, 37, 37) or grid.dtype != torch.float16 or not torch.isfinite(grid).all():
        raise ValueError('Mascaret grid failed shape/dtype/numerical checks')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    geometry = compute_dense_geometry(target_size=512, patch_size=14)
    decoder = build_decoder_for_grid(config.decoder.name, config.decoder.params,
                                    geometry=geometry, input_dim=1536, num_classes=4)
    net = SegmentationModel(decoder=decoder,
                            task_head=SegmentationHead(num_classes=4, geometry=geometry)).cuda()
    features = grid.float().unsqueeze(0).repeat(64, 1, 1, 1).cuda()
    masks = torch.arange(512 * 512, device='cuda').remainder(4).reshape(1, 512, 512).repeat(64, 1, 1)
    masks[:, -1, :] = 255
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    before = next(net.parameters()).detach().clone()
    logits = net(features).logits
    loss = net.task_head.compute_loss(logits, {'mask': masks})
    loss.backward()
    if tuple(logits.shape) != (64, 4, 512, 512) or not torch.isfinite(loss):
        raise ValueError('Decoder geometry/loss failed')
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):
        raise ValueError('Non-finite or missing decoder gradients')
    optimizer.step()
    if torch.equal(before, next(net.parameters()).detach()):
        raise ValueError('Decoder optimizer did not update parameters')
    result = dict(status='completed', feature_shape=list(grid.shape), cache_dtype=str(grid.dtype),
                  decoder_parameters=sum(p.numel() for p in net.parameters()),
                  batch_size=64, loss=loss.item(), finite_gradients=True,
                  elapsed_seconds=time.monotonic() - started)
    write_json(base / 'evidence/smoke.json', result)
    return result


def run(base: Path, job_dir: Path) -> None:
    import torch
    from soma.config import load_config
    from soma.pipeline import Pipeline
    from beetle.extract import extract_cache
    from beetle import record
    from beetle.score import score_attempts
    from beetle.comparison import main as compare

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    prep = json.loads((base / 'evidence/preparation.json').read_text())
    if prep['config_sha256'] != sha256(base / 'active-config.yaml'):
        raise ValueError('Prepared config changed')
    # save_config omits run_id, so pin it after loading or Soma starts a fresh run.
    config = replace(load_config(base / 'active-config.yaml'), run_id=RUN_ID)
    token = claim_gpu(job_dir)  # Keep this allocation alive across all phases.
    status = base / 'status.json'
    try:
        write_json(status, {'status': 'running', 'phase': 'smoke', 'job_dir': str(job_dir)})
        print(smoke(base, config), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        write_json(status, {'status': 'running', 'phase': 'extraction', 'job_dir': str(job_dir)})
        cache = extract_cache(config, base / 'extraction')
        if cache['roi_grids'] != 124697 or cache['feature_dim'] != 1536:
            raise ValueError('Full Mascaret feature cache has wrong population or channels')
        write_json(base / 'evidence/cache.json', cache)
        gc.collect()
        torch.cuda.empty_cache()
        write_json(status, {'status': 'running', 'phase': 'training', 'job_dir': str(job_dir)})
        result = Pipeline(config).run()
        record.ATTEMPTS_DIR = base / 'provenance/attempts'
        record.record_training('attempt-04', result.run_dir, config)
        rois = result.run_dir / 'segmentation_rois'
        for name, expected in prep['roi_hashes'].items():
            if sha256(rois / name) != expected:
                raise ValueError(f'Training {name} differs from baseline replay')
        write_json(status, {'status': 'running', 'phase': 'test_scoring', 'job_dir': str(job_dir)})
        score_attempts(attempts=[('attempt-04', result.run_dir)], roi_manifest=rois / 'roi_manifest.csv',
                       roi_splits=rois / 'roi_splits.csv', feature_dir=cache['feature_dir'],
                       output_dir=base / 'scores', split='test', batch_size=64, num_workers=4)
        compare(['--attempt', 'attempt-01=/maindisk/clement/beetle-test-folds-20260911/scores/attempt-01',
                 '--attempt', f'attempt-04={base / "scores/attempt-04"}',
                 '--sample-patient-csv', str(rois / 'roi_manifest.csv'), '--split', 'test',
                 '--group-csv', str(config.dataset_csv), '--group-column', 'source',
                 '--output', str(base / 'provenance/comparison.json')])
        write_json(status, {'status': 'completed', 'run_dir': str(result.run_dir),
                            'comparison': str(base / 'provenance/comparison.json')})
    except Exception as exc:
        write_json(status, {'status': 'failed', 'error': repr(exc), 'job_dir': str(job_dir)})
        raise
    finally:
        del token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--job-dir', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.base)
    elif args.job_dir is None:
        parser.error('run requires --job-dir for the GPU guard handshake')
    else:
        run(args.base, args.job_dir)


if __name__ == '__main__':
    main()
