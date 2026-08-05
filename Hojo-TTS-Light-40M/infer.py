#!/usr/bin/env python3
"""24 kHz speaker-conditioned TTS inference using exported ONNX models."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf

from onnx_model import (
    CODEC_ONNX_NAME,
    DEFAULT_MODELS_DIR,
    FINE_LOCAL_ONNX_NAME,
    LM_ONNX_NAME,
    VOICES_NPZ_NAME,
    HojoTTSLightOnnx,
)

RELEASE_ROOT = Path(__file__).resolve().parent

MODEL_BUNDLE_FILENAMES = (
    LM_ONNX_NAME,
    FINE_LOCAL_ONNX_NAME,
    CODEC_ONNX_NAME,
    VOICES_NPZ_NAME,
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))


def resolve_models_dir(repo_id: str, *, cache_dir: str | None = None) -> str:
    """Download ONNX bundle from Hugging Face and return the local directory."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=list(MODEL_BUNDLE_FILENAMES),
    )


class HojoTTSLight:
    """High-level TTS API (wraps ``HojoTTSLightOnnx``)."""

    def __init__(
        self,
        models_dir: str | Path | None = None,
        voices_npz: str | Path | None = None,
        *,
        repo_id: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if repo_id:
            models_dir = resolve_models_dir(repo_id, cache_dir=cache_dir)
        self._model = HojoTTSLightOnnx(models_dir, voices_npz=voices_npz)

    @property
    def available_voices(self) -> list[str]:
        return self._model.available_voices

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    def generate(self, text: str, *, voice: str):
        """Generate audio from text. Returns float32 waveform @ 24 kHz."""
        return self._model.generate(text, voice=voice)

    def generate_to_file(
        self,
        text: str,
        output_path: str,
        *,
        voice: str,
        sample_rate: int | None = None,
    ) -> str:
        """Generate audio and write a WAV file. Returns the output path."""
        wav = self.generate(text, voice=voice)
        if not output_path.endswith(".wav"):
            output_path += ".wav"
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sf.write(output_path, wav, sample_rate or self.sample_rate)
        return output_path


def get_model(
    model: str | Path | None = None,
    *,
    repo_id: str | None = None,
    cache_dir: str | None = None,
    voices_npz: str | Path | None = None,
) -> HojoTTSLight:
    """Load ``HojoTTSLight`` from a local models directory or Hugging Face repo.

    Args:
        model: Local directory with ONNX assets. Ignored when ``repo_id`` is set.
        repo_id: Hugging Face repository ID (e.g. ``HojoAI/Hojo-TTS-Light-40M``).
        cache_dir: Optional Hugging Face cache directory.
    """
    models_dir = None if repo_id else model
    return HojoTTSLight(
        models_dir,
        voices_npz=voices_npz,
        repo_id=repo_id,
        cache_dir=cache_dir,
    )


def _resolve_runtime_paths(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    """Return (models_dir, repo_id, cache_dir) for model loading."""
    if args.repo_id:
        return None, args.repo_id, args.cache_dir or None
    onnx_dir = args.onnx_dir or str(DEFAULT_MODELS_DIR)
    return onnx_dir, None, None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="",
        help="Hugging Face repo ID; downloads ONNX bundle via huggingface_hub",
    )
    parser.add_argument(
        "--cache-dir",
        default="",
        help="Cache directory for --repo-id downloads (optional)",
    )
    parser.add_argument(
        "--onnx_dir",
        default="",
        help=f"Local models directory (default: {DEFAULT_MODELS_DIR}; ignored if --repo-id is set)",
    )
    parser.add_argument(
        "--voices_npz",
        default="",
        help=f"voices.npz path (default: <onnx_dir>/{VOICES_NPZ_NAME})",
    )
    parser.add_argument("--text", default="", help="Target text")
    parser.add_argument("--voice", default="", help="Opaque voice ID from voices.npz (e.g. hojo_en_m_02)")
    parser.add_argument("--output_path", default="", help="Output wav path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    models_dir, repo_id, cache_dir = _resolve_runtime_paths(args)

    missing = [name for name in ("text", "voice", "output_path") if not getattr(args, name)]
    if missing:
        raise SystemExit(f"Missing required args: {', '.join(missing)}")

    tts = get_model(
        models_dir,
        repo_id=repo_id,
        cache_dir=cache_dir,
        voices_npz=args.voices_npz or None,
    )
    output_path = tts.generate_to_file(
        args.text,
        args.output_path,
        voice=args.voice,
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
