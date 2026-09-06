from types import SimpleNamespace

import numpy as np
import tifffile

from beetle.slide_predict import (
    SlideRasters,
    SlideRecord,
    export_fold,
    load_fold_slides,
    load_slide_csv,
    plan_read_spacing,
    predict_slide_mask,
    write_pyramidal_mask,
)


class ConstantPredictor:
    """Predicts class index = (x // 10) % 4 in read-pixel space; records what it saw."""

    geometry = SimpleNamespace(target_size=(16, 16))

    def __init__(self):
        self.calls = []

    def predict_array(self, rgb, *, overlap, batch_size, return_probs):
        h, w = rgb.shape[:2]
        # rgb red channel encodes the absolute read-x coordinate (see fake reader)
        xs = rgb[..., 0].astype(np.int64) + 256 * rgb[..., 1].astype(np.int64)
        self.calls.append((h, w))
        return SimpleNamespace(labels=((xs // 10) % 4).astype(np.int64))


def make_rasters(mask, *, mask_spacing=0.5, image_spacing=0.5, image_size=None, read_spacing=None):
    mask_h, mask_w = mask.shape
    read_spacing = read_spacing or image_spacing
    image_size = image_size or (mask_w, mask_h)

    def read_image(location, spacing, size):
        assert spacing == read_spacing
        w, h = size
        x0, y0 = location
        rx0 = int(round(x0 * image_spacing / read_spacing))
        xs = np.arange(rx0, rx0 + w)[None, :].repeat(h, axis=0)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = xs % 256
        rgb[..., 1] = xs // 256
        return rgb

    def read_mask(location, size):
        x0, y0 = location
        w, h = size
        return mask[y0 : y0 + h, x0 : x0 + w]

    return SlideRasters(
        image_spacing_um=image_spacing,
        image_size_wh=image_size,
        mask_spacing_um=mask_spacing,
        mask_size_wh=(mask_w, mask_h),
        request_spacing_um=read_spacing,
        read_spacing_um=read_spacing,
        read_image=read_image,
        read_mask=read_mask,
    )


def test_plan_read_spacing_follows_training_reads():
    ds = [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]
    # within tolerance: request 0.5, hs2p reads level 0 unresampled at its own pitch
    assert plan_read_spacing(level0_spacing_um=0.525, level_downsamples=ds, requested_spacing_um=0.5, tolerance=0.1) == (0.5, 0.525)
    # native_if_coarser: the three TCGA exceptions read at their native spacing
    assert plan_read_spacing(level0_spacing_um=0.657, level_downsamples=ds, requested_spacing_um=0.5, tolerance=0.1) == (0.657, 0.657)
    # far finer than the request: resampled to exactly the request
    assert plan_read_spacing(level0_spacing_um=0.25, level_downsamples=ds, requested_spacing_um=0.5, tolerance=0.1) == (0.5, 0.5)


def test_predict_slide_mask_covers_only_annotated_pixels_and_skips_empty_chunks():
    mask = np.zeros((40, 100), dtype=np.uint8)
    mask[5:30, 3:47] = 2  # annotated only in the first two 32-px chunks along x
    rasters = make_rasters(mask)
    predictor = ConstantPredictor()
    canvas = np.zeros(mask.shape, dtype=np.uint8)

    stats = predict_slide_mask(
        predictor, rasters, canvas=canvas, chunk_px=32
    )

    inside = mask > 0
    assert np.all(canvas[~inside] == 0)
    xs = np.arange(100)[None, :].repeat(40, axis=0)
    expected = ((xs // 10) % 4 + 1).astype(np.uint8)
    assert np.array_equal(canvas[inside], expected[inside])
    # only the two 32-px chunks at rows [0, 32) x cols [0, 64) hold annotation
    assert len(predictor.calls) == 2
    assert stats["annotated_pixels"] == int(inside.sum())
    assert stats["annotated_non_invasive_epithelium"] == int(inside.sum())
    assert sum(stats[f"predicted_{n}"] for n in ("other", "non_invasive_epithelium", "invasive_epithelium", "necrosis")) == int(inside.sum())
    assert stats["read_spacing_um"] == 0.5


def test_predict_slide_mask_resamples_when_mask_is_coarser_than_the_read():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:, :] = 1
    # mask at 1.0 um/px over a 0.5 um/px slide: 20 mask px == 40 read px
    rasters = make_rasters(mask, mask_spacing=1.0, image_spacing=0.5, image_size=(40, 40))
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    predict_slide_mask(
        ConstantPredictor(), rasters, canvas=canvas, chunk_px=64
    )
    # read-x = 2 * mask-x, so class = (2x // 10) % 4 -> label + 1
    xs = np.arange(20)[None, :].repeat(20, axis=0)
    assert np.array_equal(canvas, (((2 * xs) // 10) % 4 + 1).astype(np.uint8))


def test_write_pyramidal_mask_roundtrips_geometry_and_spacing(tmp_path):
    canvas = (np.arange(600 * 700, dtype=np.int64) % 5).astype(np.uint8).reshape(600, 700)
    out = tmp_path / "slide.tif"
    write_pyramidal_mask(canvas, out, spacing_um=0.5, tile=256)
    with tifffile.TiffFile(out) as tif:
        page = tif.pages[0]
        assert page.shape == (600, 700)
        assert page.is_tiled
        assert np.array_equal(page.asarray(), canvas)
        x_res = page.tags["XResolution"].value
        assert abs(x_res[0] / x_res[1] - 20_000.0) < 1e-6  # 0.5 um/px = 20000 px/cm
        assert page.tags["ResolutionUnit"].value == 3
        levels = tif.series[0].levels
        assert [lvl.shape for lvl in levels] == [(600, 700), (300, 350), (150, 175)]


def test_load_fold_slides_selects_the_manifest_validation_fold(tmp_path):
    import pytest

    dataset = tmp_path / "dataset.csv"
    dataset.write_text(
        "sample_id,image_path,label_mask_path,patient_id,validation_fold,spacing_at_level_0\n"
        "a,/x/a.tif,/x/a_mask.tif,p1,fold0,\n"
        "b,/x/b.tif,/x/b_mask.tif,p2,fold1,0.657\n"
    )
    (rec,) = load_fold_slides(dataset, 1)
    assert rec == SlideRecord("b", "p2", rec.image_path, rec.label_mask_path, 0.657)
    assert str(rec.image_path) == "/x/b.tif"
    assert [r.sample_id for r in load_fold_slides(dataset, 0)] == ["a"]
    with pytest.raises(ValueError):
        load_fold_slides(dataset, 3)


def test_export_fold_writes_masks_and_summary_and_resumes(tmp_path):
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[:15] = 3
    record = SlideRecord("s1", "p1", tmp_path / "s1.tif", tmp_path / "s1_mask.tif", None)
    opened = []

    def open_rasters(rec):
        opened.append(rec.sample_id)
        return make_rasters(mask)

    kwargs = dict(
        predictor=ConstantPredictor(),
        records=(record,),
        fold=2,
        output_dir=tmp_path / "out",
        open_rasters=open_rasters,
        chunk_px=64,
    )
    (path,) = export_fold(**kwargs)
    assert path == tmp_path / "out" / "fold_2" / "s1.tif"
    written = tifffile.imread(path)
    assert written.shape == mask.shape and np.all(written[15:] == 0) and np.all(written[:15] > 0)
    summary = (tmp_path / "out" / "fold_2" / "summary.csv").read_text().splitlines()
    assert summary[0].startswith("sample_id,patient_id,fold,")
    assert summary[1].startswith("s1,p1,2,")
    assert f",{15 * 30}," in summary[1]
    assert not list((tmp_path / "out" / "fold_2").glob(".canvas-*"))

    export_fold(**kwargs)  # resumes: does not reopen the slide
    assert opened == ["s1"]


def test_load_slide_csv_defaults_sample_id_to_the_wsi_stem(tmp_path):
    listing = tmp_path / "slides.csv"
    listing.write_text(
        "wsi_path,roi_mask_path\n"
        "/a/images/202B.tif,/a/roi-masks/202B_roi_mask.tif\n"
        "/a/images/TCGA-D8-A27G-01Z-00-DX1.04FC.tif,/a/roi-masks/TCGA-D8-A27G-01Z-00-DX1.04FC_roi_mask.tif\n"
    )
    records = load_slide_csv(listing)
    assert [r.sample_id for r in records] == ["202B", "TCGA-D8-A27G-01Z-00-DX1.04FC"]
    assert records[0].label_mask_path.name == "202B_roi_mask.tif"
    assert records[0].spacing_at_level_0 is None and records[0].patient_id == ""


def test_write_pyramidal_mask_can_write_deflate_without_imagecodecs(tmp_path):
    canvas = np.zeros((300, 300), dtype=np.uint8)
    canvas[:100] = 4
    out = tmp_path / "deflate.tif"
    write_pyramidal_mask(canvas, out, spacing_um=0.5, tile=256, compression="zlib")
    with tifffile.TiffFile(out) as tif:
        assert tif.pages[0].compression.name in ("ADOBE_DEFLATE", "DEFLATE")
        assert np.array_equal(tif.pages[0].asarray(), canvas)


def test_candidate_chunks_dilates_the_overview_and_covers_borders():
    from beetle.slide_predict import candidate_chunks

    overview = np.zeros((10, 10), dtype=np.uint8)
    overview[4, 4] = 1  # level-0 pixel (16, 16) with downsample 4 -> chunk (0, 0) plus dilation
    chunks = candidate_chunks(overview, 4.0, mask_size_wh=(40, 40), chunk_px=16)
    assert chunks == {(0, 0), (16, 0), (0, 16), (16, 16)}
    assert candidate_chunks(np.zeros((10, 10), np.uint8), 4.0, mask_size_wh=(40, 40), chunk_px=16) == set()
    # overview pixels at the far edge clamp into the last chunk rather than past it
    overview[:] = 0
    overview[9, 9] = 3
    assert candidate_chunks(overview, 4.0, mask_size_wh=(40, 40), chunk_px=16) == {(32, 32)}


def test_predict_slide_mask_uses_the_overview_to_avoid_reading_empty_chunks():
    mask = np.zeros((64, 128), dtype=np.uint8)
    mask[40:60, 100:120] = 1  # only the chunk at (96, 32)
    rasters = make_rasters(mask)
    reads = []
    inner_read_mask = rasters.read_mask

    def counting_read_mask(location, size):
        reads.append(location)
        return inner_read_mask(location, size)

    rasters.read_mask = counting_read_mask
    rasters.read_mask_overview = lambda: (mask[::4, ::4], 4.0)
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    stats = predict_slide_mask(ConstantPredictor(), rasters, canvas=canvas, chunk_px=32)
    assert stats["annotated_pixels"] == 400
    assert np.count_nonzero(canvas) == 400
    assert (96, 32) in reads and len(reads) <= 4  # neighbours via dilation, never all 8 chunks


def test_overview_level_picks_the_coarsest_level_within_the_cap():
    from beetle.slide_predict import _overview_level

    assert _overview_level([(1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (16.0, 16.0), (32.0, 32.0)], 16.0) == 3
    assert _overview_level([(1.0, 1.0)], 16.0) == 0
    assert _overview_level([1.0, 4.0, 64.0], 16.0) == 1


def test_open_slide_rasters_stamps_exception_masks_with_the_slide_override(monkeypatch, tmp_path):
    import hs2p.wsi.masks as masks_mod
    import hs2p.wsi.wsi as wsi_mod

    from beetle.slide_predict import open_slide_rasters

    class FakeWSI:
        def __init__(self, path, backend, spacing_at_level_0=None):
            self.path = path
            declared = 0.5 if "mask" in str(path) else 0.657
            self.spacing = spacing_at_level_0 if spacing_at_level_0 is not None else declared
            self.level_dimensions = [(1000, 800), (500, 400)]
            self.level_downsamples = [(1.0, 1.0), (2.0, 2.0)]

        def get_level_spacing(self, level):
            return self.spacing * self.level_downsamples[level][0]

    seen = []

    def fake_read_label(wsi, location, spacing, size, *, tolerance):
        seen.append(spacing)
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    monkeypatch.setattr(wsi_mod, "WSI", FakeWSI)
    monkeypatch.setattr(masks_mod, "read_label_region_at_spacing", fake_read_label)
    record = SlideRecord("tcga", "p", tmp_path / "tcga.svs", tmp_path / "tcga_mask.tif", 0.657)
    rasters = open_slide_rasters(record, backend="auto", mask_backend="openslide", requested_spacing_um=0.5, tolerance=0.1)
    # output geometry is stamped with the slide's true pitch, and reads at that pitch (native_if_coarser)
    assert rasters.mask_spacing_um == 0.657 and rasters.read_spacing_um == 0.657
    # but the mask file is still read at the spacing it declares, so hs2p picks level 0 unresampled
    rasters.read_mask((0, 0), (10, 10))
    assert seen == [0.5]
