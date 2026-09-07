import argparse
from concurrent.futures import ThreadPoolExecutor
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import easyocr
import numpy as np


def tc_to_ms(value: str, fps: float) -> int:
    h, m, s, frame = (int(part) for part in value.split(":"))
    ms = (h * 3600 + m * 60 + s) * 1000 + round(frame * 1000 / fps)
    if fps % 1 != 0:
        # Subtitle Edit writes fractional-rate BDN timecodes on the nominal
        # integer grid (true milliseconds divided by 1.001); scale back to
        # wall-clock time the same way its own BdnXml reader does. Without
        # this the whole timeline runs ~0.1% early (about 7 s over 2 h).
        ms = round(ms * 1.001)
    return ms


def srt_time(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def split_lines(path: Path) -> list[np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        rgb = image[:, :, :3]
        gray = cv2.cvtColor(
            (rgb * (alpha[:, :, None].astype(np.float32) / 255)).astype(np.uint8),
            cv2.COLOR_BGR2GRAY,
        )
        mask = alpha > 8
    elif image.ndim == 3 and image.shape[2] == 2:
        alpha = image[:, :, 1]
        gray = (image[:, :, 0] * (alpha.astype(np.float32) / 255)).astype(np.uint8)
        mask = alpha > 8
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray > 8
    else:
        gray = image
        mask = gray > 8
    occupied_rows = np.flatnonzero(mask.any(axis=1))
    if not len(occupied_rows):
        return []
    row_groups = np.split(
        occupied_rows, np.flatnonzero(np.diff(occupied_rows) > 5) + 1
    )
    crops = []
    for rows in row_groups:
        y0 = max(0, int(rows[0]) - 3)
        y1 = min(gray.shape[0], int(rows[-1]) + 4)
        occupied_columns = np.flatnonzero(mask[y0:y1].any(axis=0))
        if not len(occupied_columns):
            continue
        x0 = max(0, int(occupied_columns[0]) - 6)
        x1 = min(gray.shape[1], int(occupied_columns[-1]) + 7)
        crop = gray[y0:y1, x0:x1]
        if crop.shape[0] >= 10 and crop.shape[1] >= 10:
            crops.append(crop)
    return crops


def recognize_batches(reader, line_items, batch_size: int, pack_size: int):
    recognized = []
    for offset in range(0, len(line_items), pack_size):
        chunk = line_items[offset : offset + pack_size]
        width = max(item[2].shape[1] for item in chunk) + 16
        height = sum(item[2].shape[0] + 8 for item in chunk) + 8
        canvas = np.zeros((height, width), dtype=np.uint8)
        boxes = []
        y = 8
        for _, _, crop in chunk:
            h, w = crop.shape
            canvas[y : y + h, 8 : 8 + w] = crop
            boxes.append([8, 8 + w, y, y + h])
            y += h + 8
        results = reader.recognize(
            canvas,
            horizontal_list=boxes,
            free_list=[],
            decoder="greedy",
            beamWidth=1,
            batch_size=batch_size,
            workers=0,
            detail=1,
            blocklist="|_~",
            paragraph=False,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            reformat=False,
        )
        if len(results) != len(chunk):
            raise RuntimeError(
                f"EasyOCR returned {len(results)} results for {len(chunk)} packed "
                "lines; aborting instead of misaligning cues"
            )
        for (event_index, line_index, _), result in zip(chunk, results):
            _, text, confidence = result
            recognized.append(
                (event_index, line_index, text.strip(), float(confidence))
            )
        print(
            f"OCR lines {min(offset + pack_size, len(line_items))}/{len(line_items)}",
            flush=True,
        )
    return recognized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a BDN XML/PNG directory to SRT using batched CUDA EasyOCR."
    )
    parser.add_argument("bdn_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip the first N events; pair with --limit to benchmark a "
        "mid-film sample (cue numbers restart at 1 for the sample)",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ocr-batch", type=int, default=64)
    parser.add_argument("--pack-lines", type=int, default=128)
    parser.add_argument("--io-workers", type=int, default=8)
    args = parser.parse_args()

    index_path = args.bdn_dir / "index.xml"
    if not index_path.is_file():
        raise SystemExit(f"Missing BDN index: {index_path}")

    root = ET.parse(index_path).getroot()
    format_node = root.find("./Description/Format")
    if format_node is None or "FrameRate" not in format_node.attrib:
        raise SystemExit("BDN index does not declare FrameRate")
    fps = float(format_node.attrib["FrameRate"])
    events = root.findall("./Events/Event")
    if args.offset:
        events = events[args.offset :]
    if args.limit:
        events = events[: args.limit]

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    if str(reader.device).lower() == "cpu":
        raise SystemExit("EasyOCR initialized on CPU; fix PyTorch CUDA before continuing")

    started = time.time()
    jobs = []
    for event_index, event in enumerate(events):
        for graphic in event.findall("Graphic"):
            if graphic.text:
                jobs.append((event_index, args.bdn_dir / graphic.text.strip()))
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        crop_lists = list(pool.map(lambda job: split_lines(job[1]), jobs))

    line_items = []
    line_counters = [0] * len(events)
    for (event_index, _), crops in zip(jobs, crop_lists):
        for crop in crops:
            line_items.append((event_index, line_counters[event_index], crop))
            line_counters[event_index] += 1
    if not line_items:
        raise SystemExit("No text line images were detected")

    recognized = recognize_batches(
        reader, line_items, args.ocr_batch, args.pack_lines
    )
    event_texts = [[] for _ in events]
    low_confidence = []
    for event_index, line_index, text, confidence in recognized:
        event_texts[event_index].append((line_index, text))
        if confidence < 0.45:
            low_confidence.append((event_index + 1, confidence, text))

    blocks = []
    for cue, (event, text_items) in enumerate(zip(events, event_texts), 1):
        content = "\n".join(text for _, text in sorted(text_items))
        start = srt_time(tc_to_ms(event.attrib["InTC"], fps))
        end = srt_time(tc_to_ms(event.attrib["OutTC"], fps))
        blocks.append(f"{cue}\n{start} --> {end}\n{content}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(blocks), encoding="utf-8-sig")
    qc_path = args.output.with_suffix(".low-confidence.tsv")
    qc_path.write_text(
        "cue\tconfidence\ttext\n"
        + "\n".join(
            f"{cue}\t{confidence:.4f}\t{text}"
            for cue, confidence, text in low_confidence
        ),
        encoding="utf-8-sig",
    )
    print(
        f"Done: {len(events)} cues, {len(line_items)} lines, "
        f"{time.time() - started:.1f}s, {len(low_confidence)} low-confidence",
        flush=True,
    )


if __name__ == "__main__":
    main()
