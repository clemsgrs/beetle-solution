# Algorithmic description

## Development cohort and splits

The development cohort comprised 587 slides from 527 patients. We used the five organizer-provided patient folds. For fold `k`, organizer fold `k` was held out for evaluation, fold `(k + 1) mod 5` was used for validation, and the remaining three folds were used for training. No External labels were used for model selection.

Training examples were non-overlapping 512×512-pixel regions of interest (ROIs) sampled at 0.5 µm/px when a matching resolution was available. A 10% relative spacing tolerance was used; three coarser slides at approximately 0.6575 µm/px were processed at their native resolution rather than upsampled. An ROI was retained when at least one annotated class covered 5% of its pixels, producing 124,697 ROIs. Raw label 0 was ignored; labels 1–4 represented Other, non-invasive epithelium, invasive epithelium, and necrosis.

## Frozen encoder and decoder

Each 512×512 ROI was reflect-padded by 6 pixels along its bottom and right edges. The resulting 518×518 input is the smallest square at least 512 pixels wide whose dimensions are divisible by Virchow2's 14-pixel patch size. Frozen Virchow2 patch tokens were extracted with 224×224 encoder windows and 50% overlap. For a 518-pixel input this gives four window starts per axis; raised-cosine blending combined token predictions in overlapping windows into a 37×37×1,280 grid. Cached grids were stored in FP16 and converted to FP32 for decoder training.

Each fold trained a 1,510,660-parameter decoder comprising a 1×1 projection from 1,280 to 256 channels followed by GroupNorm (32 groups) and ReLU; two blocks of 2× bilinear upsampling, 3×3 convolution, GroupNorm, and ReLU; and a 1×1 four-class classifier. The decoder produced 148×148×4 logits. These logits were bilinearly interpolated to 518×518×4, and only the top-left 512×512 region corresponding to the original, unpadded ROI was retained for loss computation and prediction.

Attempt 02 was a development-only ablation that reused the verified Attempt 01 features and protocol, changing only the decoder depth from two to four upsampling/convolution blocks. Preflight selected physical batch size 64 with one accumulation step and froze it for all five folds. Its equal-weight five-fold dataset-global mean class Dice was 0.8880192 ± 0.0286550 (sample standard deviation), compared only against Attempt 01's 0.8861255 ± 0.0284992. This result did not select or replace an External model.

We minimized unweighted pixelwise cross-entropy plus multiclass soft-Dice, each with coefficient 1, over annotated pixels. Optimization used Adam with learning rate 1×10⁻⁴ and weight decay 1×10⁻⁵, a 30-epoch cosine schedule, batch size 64, no data augmentation, and fold seeds 0–4. Training stopped after eight epochs without improvement in validation dataset-global mean class Dice; the best checkpoint on this criterion was retained.

## External inference

Training-batch ROIs were sampled uniformly. The submitted configuration achieved a mean fold-macro dataset-global mean class Dice of 0.8861255 over the five development folds.

The official External data consisted of pre-extracted ROI PNGs of varying dimensions. We slid 512×512 model tiles over each full ROI with 50% overlap. This is the outer segmentation tiling step: each such model tile is still reflect-padded to 518×518 and encoded internally with the overlapping 224×224 Virchow2 windows described above. We averaged the five fold-specific per-pixel softmax tensors and Hann-blended probabilities where outer tiles overlapped. The maximum-probability class was mapped to submission labels 1–4.

Each output was written as a single-channel PNG with exactly the input ROI's filename, dimensions, and pixel alignment. The produced submission contained 170 masks, passed the filename/mode/dimension/label contract, and was packaged as a flat ZIP archive.
