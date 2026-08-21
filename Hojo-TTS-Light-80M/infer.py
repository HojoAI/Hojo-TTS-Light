#!/usr/bin/env python3
"""Hojo-TTS-Light ONNX inference: ref wav + ref text → target text @ 24 kHz."""

from __future__ import annotations

import argparse
import os

import soundfile as sf

from onnx_model import (
    DEFAULT_MODELS_DIR,
    HojoTTSLightOnnx,
    configure_cpu_threads,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx_dir",
        default="",
        help=f"Local models directory (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--prompt_speech", default="", help="Reference wav path")
    parser.add_argument("--prompt_text", default="", help="Reference transcript")
    parser.add_argument("--text", default="", help="Target text")
    parser.add_argument("--output_path", default="", help="Output wav path")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    missing = [
        name
        for name in ("prompt_speech", "text", "output_path")
        if not getattr(args, name)
    ]
    if missing:
        raise SystemExit(f"Missing required args: {', '.join(missing)}")

    if args.num_threads > 0:
        configure_cpu_threads(args.num_threads)

    onnx_dir = args.onnx_dir or str(DEFAULT_MODELS_DIR)
    tts = HojoTTSLightOnnx(
        onnx_dir,
        provider=args.provider,
        device_id=args.device_id,
        num_threads=args.num_threads,
    )
    wav = tts.generate(
        args.text,
        prompt_speech=args.prompt_speech,
        prompt_text=args.prompt_text,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )

    output_path = (
        args.output_path
        if args.output_path.endswith(".wav")
        else f"{args.output_path}.wav"
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sf.write(output_path, wav, tts.sample_rate)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
