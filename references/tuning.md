# CUDA batching and pipeline tuning

## The important batching distinction

EasyOCR's `batch_size` does not create useful cross-image batching when code calls `readtext()` separately for every PNG. That pattern repeatedly invokes detection and sends tiny recognition jobs to the GPU.

For uniform Blu-ray subtitle graphics:

1. Read PNG files concurrently on CPU.
2. Use the alpha channel to find occupied rows and split each graphic into line crops.
3. Pack many crops into one grayscale canvas and supply their rectangles as `horizontal_list` to `Reader.recognize()`.
4. Set `batch_size` on that recognition call so many cropped lines are evaluated together.

This removes neural text detection from the hot path and creates a real recognizer batch.

## Starting values

| GPU memory | `--ocr-batch` | `--pack-lines` | `--io-workers` |
|---:|---:|---:|---:|
| 4 GB | 32 | 64 | 4–8 |
| 8 GB | 128 | 256 | 8–12 |
| 12–16 GB | 256 | 512 | 12–16 |

Reduce `--ocr-batch` first after CUDA out-of-memory errors. `--pack-lines` controls how many crops share one recognition call; it can be larger than the CUDA batch.

## Concurrency strategy

- Use one GPU OCR process per GPU. Multiple independent OCR processes duplicate model memory and commonly reduce throughput.
- Increase batching inside that process instead of launching one process per movie.
- Overlap GPU OCR for movie N with sequential PGS extraction for movie N+1 when storage bandwidth permits.
- On one HDD or a constrained network share, keep only one large MKV reader active. Parallel reads can turn sequential I/O into slower random I/O.
- PNG decoding is CPU work; `--io-workers` overlaps it with minimal memory cost until storage or CPU becomes saturated.
- Do not emit high-frequency ffmpeg progress into an abandoned pipe. Use quiet logging for background pipelines or keep consuming stdout.

## Benchmark method

Before processing a collection, benchmark a representative 5–10 minute sample and compare cues and lines per second, peak GPU memory, empty cue count, low-confidence count, and visible errors in names, punctuation, italics, and speaker labels.

Sample from mid-film, not the opening: use `--offset N --limit N` on the full BDN folder (N counts events) instead of `--limit` alone, so the sample includes dense dialogue rather than studio logos. After benchmarking, verify timestamps against the video with one known spoken line — the frame-rate conversion includes the ×1.001 SMPTE adjustment for fractional rates, and a single spot-check confirms there is no cumulative drift.

Use the fastest configuration that preserves acceptable text quality. Specialized generative vision OCR can be useful for difficult low-confidence cues, but applying it to every clean PGS cue is usually far slower than batched recognizer OCR.
