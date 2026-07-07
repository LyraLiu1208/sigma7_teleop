#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
import time

import cv2
import numpy as np


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return b""
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Display a live PNG-compressed RGB stream in an OpenCV window.")
    parser.add_argument("--title", type=str, default="eye_in_hand_rgb")
    parser.add_argument(
        "--report-keys",
        action="store_true",
        help="Emit key codes to stdout so a parent process can react to keyboard input.",
    )
    args = parser.parse_args(argv)

    try:
        cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)
    except Exception as exc:
        print(f"failed to create OpenCV window: {exc}", file=sys.stderr, flush=True)
        return 1

    latest_frame = None
    while True:
        header = _read_exact(sys.stdin.buffer, 4)
        if not header:
            break
        (payload_size,) = struct.unpack("!I", header)
        if payload_size == 0:
            break
        payload = _read_exact(sys.stdin.buffer, payload_size)
        if len(payload) != payload_size:
            break
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            continue
        latest_frame = frame
        cv2.imshow(args.title, latest_frame)
        key = cv2.waitKeyEx(1)
        if args.report_keys and key != -1:
            print(key, flush=True)
        if key == 27:
            break
        time.sleep(0.001)

    try:
        cv2.destroyWindow(args.title)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
