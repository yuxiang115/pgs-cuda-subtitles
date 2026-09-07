# pgs-cuda-subtitles

An agent skill for turning Blu-ray PGS (`hdmv_pgs_subtitle`) bitmap subtitle tracks in MKV files into accurate English SRT files with true batched CUDA OCR — without re-encoding or remuxing the source video — and optionally producing a corrected English SRT and a Simplified Chinese translation.

## What it does

1. Route the source with `ffprobe`: text tracks are extracted directly; only `hdmv_pgs_subtitle` enters the OCR path.
2. Losslessly extract the selected PGS stream to `.sup` with `ffmpeg`, then convert it to a BDN XML folder (`index.xml` + numbered PNGs) with Subtitle Edit's headless `seconv` CLI.
3. OCR the BDN folder with `scripts/ocr_bdn_cuda.py`:
   - alpha-channel line segmentation (no neural text detection in the hot path),
   - parallel PNG decoding,
   - many line crops packed into a single `Reader.recognize()` call with a real CUDA `batch_size`,
   - BDN timecodes converted with the declared frame rate, including the ×1.001 SMPTE adjustment Subtitle Edit uses for fractional rates (23.976/29.97),
   - a `*.low-confidence.tsv` report for the correction pass.
4. Correct and translate in contextual batches with stable cue IDs; reconstruct SRT files from the original cue numbers and timecodes instead of trusting a model to reproduce them.

## Requirements

- Python with `torch` (CUDA), `easyocr`, `opencv-python`, `numpy`
- `ffmpeg` / `ffprobe`
- A recent [Subtitle Edit](https://github.com/SubtitleEdit/subtitleedit) build providing `seconv` (image-to-image conversion must use the preserve-bitmaps path)

## Usage (agents)

Install the skill, then follow `SKILL.md`. For an 8 GB GPU:

```powershell
python scripts/ocr_bdn_cuda.py "work/movie-bdn" "movie.en.ocr.srt" `
  --ocr-batch 128 --pack-lines 256 --io-workers 12
```

Benchmark with `--offset N --limit N` from mid-film before batch-processing a collection. See `references/tuning.md` for batch sizing per GPU memory tier and pipeline/concurrency strategy.

## Install

```bash
npx skills add yuxiang115/pgs-cuda-subtitles -g
```
