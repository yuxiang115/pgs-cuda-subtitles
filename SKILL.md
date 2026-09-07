---
name: pgs-cuda-subtitles
description: Extract embedded Blu-ray PGS subtitle tracks from MKV files, convert them to timed text with true batched CUDA OCR, and optionally produce corrected English and translated Simplified Chinese SRT files. Use for bitmap subtitle tracks such as hdmv_pgs_subtitle; do not use OCR when the source already contains a text subtitle track or when subtitles are burned into video frames.
---

# PGS CUDA Subtitles

Build accurate SRT subtitles without modifying the source video. Prefer a resource pipeline that overlaps sequential media extraction, CPU image preparation, GPU OCR, and batched translation.

## Route the source correctly

1. Inspect subtitle streams with `ffprobe` and capture codec, language, title, disposition, duration, and packet count when available.
2. If the chosen subtitle is text (`subrip`, `ass`, `webvtt`, etc.), extract it directly and skip this skill's OCR path.
3. For `hdmv_pgs_subtitle`, identify the requested full-dialogue track. Do not assume every English track is dialogue: later tracks are often commentary or SDH variants. Use metadata and brief samples; surface the choice when it remains ambiguous.
4. Keep source MKV files read-only. Write intermediates to a dedicated work directory and final `.srt` files beside the video or to the user's requested output directory.

## Extract and decode PGS

Extract only the selected stream, without transcoding:

```powershell
ffmpeg -y -v error -i "movie.mkv" -map 0:2 -c:s copy "work/movie.eng.sup"
```

Convert the SUP to a BDN XML folder (numbered PNG images plus `index.xml`) with Subtitle Edit's `seconv` command line tool:

```powershell
seconv.exe "work/movie.eng.sup" bdnxml `
  --output-folder:"work/movie-bdn" `
  --overwrite
```

Requirements and caveats:

- Use a recent Subtitle Edit / seconv build. Image-to-image conversion (SUP → BDN-XML) must take the preserve-bitmaps path, not OCR-plus-rerender; a quick check is that the conversion succeeds with no OCR engine installed (older builds failed with "Tesseract not found on PATH" or silently re-rendered OCR text).
- `--output-filename` is not honored for the folder-based BDN-XML target. Locate the directory that actually contains `index.xml` (e.g. `Get-ChildItem -Recurse -Filter index.xml`) and pass that directory to the OCR script.
- BDN `InTC`/`OutTC` carry PGS timing on the frame grid declared by `Description/Format@FrameRate` in `index.xml`. `scripts/ocr_bdn_cuda.py` applies the ×1.001 SMPTE adjustment automatically for fractional rates (23.976/29.97), matching Subtitle Edit's own BDN XML reader/writer convention.

## Use true CUDA batching

Verify CUDA before a full run:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda, torch.cuda.get_device_name(0))"
```

Run [scripts/ocr_bdn_cuda.py](scripts/ocr_bdn_cuda.py). For an 8 GB GPU, start with:

```powershell
python scripts/ocr_bdn_cuda.py "work/movie-bdn" "movie.en.ocr.srt" `
  --ocr-batch 128 --pack-lines 256 --io-workers 12
```

The positional BDN argument is the directory containing `index.xml`. For a speed and accuracy benchmark sample, skip the opening logos with `--offset N` (events skipped) and cap with `--limit N`; cue numbers in the sample output restart at 1.

The script deliberately avoids calling `Reader.readtext()` once per subtitle. It uses alpha-based line segmentation, parallel image loading, packs many line crops into one recognition call, and lets EasyOCR's recognizer process a real CUDA batch. Review [references/tuning.md](references/tuning.md) before changing batch or concurrency settings.

## Correct and translate

Treat OCR output as a draft. Preserve a raw `*.en.ocr.srt`, then create a corrected `*.en.srt` and optional `*.zh-CN.srt`.

- Prioritize cues listed in `*.low-confidence.tsv`, plus empty cues and text containing unlikely symbols.
- Correct common confusions such as `I`/`l`, stray `_`/`~`, missing spaces, doubled letters, and punctuation artifacts. Never change timing merely to fix text.
- Translate in contextual batches of roughly 30–80 cues. Use stable cue IDs or JSON records, never raw SRT blocks as an unconstrained generation target.
- Require exactly one corrected-English and one translated-Chinese value for every input cue ID. Retry only missing or invalid batches.
- Apply a title-specific glossary consistently. For established films, prefer the region/language convention requested by the user.
- Reconstruct SRT files from the original cue numbers and timecodes rather than trusting a model to reproduce them.

Translation concurrency depends on the backend. A single local GPU model often performs best with one loaded model and a small number of concurrent requests; benchmark 1, 2, and 4 workers instead of assuming more processes are faster.

## Validate before delivery

Check these invariants programmatically:

- BDN event count equals raw OCR SRT cue count.
- Raw English, corrected English, and Chinese files have identical cue numbers and timestamps.
- No cue is silently missing, duplicated, or empty unless the source event is genuinely non-textual.
- Timestamps are ordered and parseable; source frame rate was used for frame-to-millisecond conversion, including the ×1.001 adjustment for fractional rates. Spot-check one cue's timestamp against the corresponding PGS display time (or a known spoken line) to confirm zero cumulative drift.
- UTF-8 encoding is used; UTF-8 BOM is acceptable for broad Windows player compatibility.

Report the selected source stream, OCR backend and CUDA status, cue count, low-confidence count, output paths, and any remaining uncertainty.
