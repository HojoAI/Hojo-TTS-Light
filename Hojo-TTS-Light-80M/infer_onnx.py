#!/usr/bin/env python3
"""ONNX inference: local Hojo-TTS-Light-llm.onnx (KV-cache) + Hojo codec.

Expected layout::

    onnx/
        Hojo-TTS-Light-llm.onnx
        config.json
        generation_config.json    # optional
        tokenizer.json
        tokenizer_config.json
        Hojo-TTS-Light-encoder.onnx
        Hojo-TTS-Light-decoder.onnx
"""

from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
import transformers
from optimum.onnxruntime import ORTModelForCausalLM

SPEECH_TOKEN_PATTERN = "[{}]"
SPEECH_START_TOKEN = "[speech_start]"
SPEECH_END_TOKEN = "[speech_end]"
REF_TEXT_START_TOKEN = "[ref_text_start]"
REF_TEXT_END_TOKEN = "[ref_text_end]"
TARGET_TEXT_START_TOKEN = "[target_text_start]"
TARGET_TEXT_END_TOKEN = "[target_text_end]"
REF_SPEECH_START_TOKEN = "[ref_speech_start]"
REF_SPEECH_END_TOKEN = "[ref_speech_end]"
TARGET_SPEECH_START_TOKEN = "[target_speech_start]"
TARGET_SPEECH_END_TOKEN = "[target_speech_end]"
CODEC_SAMPLE_RATE = 16000

_HOJO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX_DIR = os.path.join(_HOJO_ROOT, "onnx")
_LLM_ONNX_NAME = "Hojo-TTS-Light-llm.onnx"
_ONNX_REQUIRED = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "Hojo-TTS-Light-encoder.onnx",
    "Hojo-TTS-Light-decoder.onnx",
)
_AUDIO_TOKEN_REGEX = re.compile(r"^\[(\d+)\]$")


def _validate_onnx_dir(onnx_dir: str) -> None:
    """Ensure onnx/ contains LM, tokenizer, and codec assets."""
    llm_onnx = os.path.join(onnx_dir, _LLM_ONNX_NAME)
    if not os.path.isfile(llm_onnx):
        raise FileNotFoundError(f"missing LM ONNX: {llm_onnx}")

    for name in _ONNX_REQUIRED:
        path = os.path.join(onnx_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing onnx resource: {path}")


def _resolve_generation_tokens(tokenizer: transformers.PreTrainedTokenizerBase):
    ref_text_start = REF_TEXT_START_TOKEN
    ref_text_end = REF_TEXT_END_TOKEN
    target_text_start = TARGET_TEXT_START_TOKEN
    target_text_end = TARGET_TEXT_END_TOKEN
    target_speech_end = TARGET_SPEECH_END_TOKEN
    if tokenizer.convert_tokens_to_ids(target_speech_end) == tokenizer.unk_token_id:
        target_speech_end = SPEECH_END_TOKEN
    return (
        ref_text_start,
        ref_text_end,
        target_text_start,
        target_text_end,
        target_speech_end,
    )


def _build_audio_token_id_to_code_table(
    tokenizer: transformers.PreTrainedTokenizerBase,
    device: torch.device,
) -> torch.Tensor:
    vocab_size = len(tokenizer)
    table_cpu = torch.full((vocab_size,), -1, dtype=torch.long)
    for tid in range(vocab_size):
        token_str = tokenizer.decode([tid], skip_special_tokens=False).strip()
        match = _AUDIO_TOKEN_REGEX.match(token_str)
        if match is not None:
            table_cpu[tid] = int(match.group(1))
    return table_cpu.to(device=device)


def _extract_audio_codes_from_generated_tail(
    generated_tail_ids: torch.Tensor,
    speech_end_id: int,
    id_to_code: torch.Tensor,
) -> torch.Tensor:
    if generated_tail_ids.numel() == 0:
        return generated_tail_ids.new_empty((0,))
    ends = (generated_tail_ids == speech_end_id).nonzero(as_tuple=False)
    seq = generated_tail_ids[: int(ends[0].item())] if ends.numel() > 0 else generated_tail_ids
    if seq.numel() == 0:
        return generated_tail_ids.new_empty((0,))
    codes = id_to_code[seq]
    return codes[codes >= 0]


def _onnx_input_type_to_numpy_dtype(type_str: str) -> np.dtype:
    if "float16" in type_str:
        return np.float16
    if "float" in type_str:
        return np.float32
    raise ValueError(f"Unsupported encoder input type: {type_str}")


def _load_wav(path: str, target_sample_rate: int) -> tuple[torch.Tensor, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav.astype(np.float32, copy=False))
    if sr != target_sample_rate:
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sample_rate)
        sr = target_sample_rate
    return wav_t, sr


def _fit_wav_to_onnx_encoder_time(wav_np: np.ndarray, enc_time: int | None) -> np.ndarray:
    if enc_time is None:
        return wav_np
    cur_t = int(wav_np.shape[-1])
    if cur_t == enc_time:
        return wav_np
    if cur_t > enc_time:
        return wav_np[..., :enc_time]
    pad_t = enc_time - cur_t
    return np.pad(wav_np, ((0, 0), (0, 0), (0, pad_t)), mode="edge")


def build_prompt(tokenizer, ref_text: str, target_text: str, prompt_audio_tokens_str: str):
    (
        ref_text_start_token,
        ref_text_end_token,
        target_text_start_token,
        target_text_end_token,
        target_speech_end_token,
    ) = _resolve_generation_tokens(tokenizer)

    ref_speech_start_token = REF_SPEECH_START_TOKEN
    ref_speech_end_token = REF_SPEECH_END_TOKEN
    target_speech_start_token = TARGET_SPEECH_START_TOKEN
    if tokenizer.convert_tokens_to_ids(target_speech_start_token) == tokenizer.unk_token_id:
        ref_speech_start_token = SPEECH_START_TOKEN
        ref_speech_end_token = SPEECH_END_TOKEN
        target_speech_start_token = SPEECH_START_TOKEN

    prompt = (
        f"{ref_text_start_token}{ref_text}{ref_text_end_token} "
        f"{target_text_start_token}{target_text}{target_text_end_token}"
        f"{ref_speech_start_token}{prompt_audio_tokens_str}{ref_speech_end_token}"
        f"{target_speech_start_token}"
    )
    return prompt, target_speech_end_token


def decode_codes_with_onnx_decoder(
    sess_dec: ort.InferenceSession,
    codes: torch.Tensor,
    device: torch.device,
    fixed_chunk_tokens: int | None,
) -> np.ndarray:
    if codes.numel() == 0:
        return np.zeros((0,), dtype=np.float32)

    dec_input = sess_dec.get_inputs()[0]
    dec_input_name = dec_input.name
    dec_input_type = dec_input.type
    decoder_uses_codes = ("int64" in dec_input_type) or (dec_input_name == "vq_codes")
    if not decoder_uses_codes:
        raise RuntimeError(
            "This script only supports ONNX exports where decoder input is int64 vq_codes."
        )

    if fixed_chunk_tokens is None:
        q_idx = codes.to(dtype=torch.long, device=device).view(1, 1, -1).cpu().numpy().astype(np.int64, copy=False)
        return sess_dec.run(None, {dec_input_name: q_idx})[0][0, 0]

    chunk_tokens = int(fixed_chunk_tokens)
    context_tokens = max(1, min(8, chunk_tokens // 4))
    hop_tokens = max(1, chunk_tokens - 2 * context_tokens)
    wav_chunks: list[np.ndarray] = []
    total = int(codes.numel())
    core_start = 0

    while core_start < total:
        core_end = min(core_start + hop_tokens, total)
        core_len = core_end - core_start

        desired_start = core_start - context_tokens
        desired_end = desired_start + chunk_tokens
        src_l = max(0, desired_start)
        src_r = min(total, desired_end)
        cur = codes[src_l:src_r].contiguous()

        left_pad = max(0, -desired_start)
        right_pad = max(0, desired_end - total)
        if left_pad > 0:
            cur = torch.cat([cur[:1].repeat(left_pad), cur], dim=0)
        if right_pad > 0:
            cur = torch.cat([cur, cur[-1:].repeat(right_pad)], dim=0)
        if int(cur.numel()) != chunk_tokens:
            pad_need = chunk_tokens - int(cur.numel())
            cur = torch.cat([cur, cur[-1:].repeat(pad_need)], dim=0)

        q_idx = cur.to(dtype=torch.long, device=device).view(1, 1, -1).cpu().numpy().astype(np.int64, copy=False)
        wav_chunk = sess_dec.run(None, {dec_input_name: q_idx})[0][0, 0]

        samples_per_token = max(1, int(wav_chunk.shape[0]) // chunk_tokens)
        core_l = context_tokens * samples_per_token
        core_r = core_l + core_len * samples_per_token
        wav_chunks.append(wav_chunk[core_l:core_r])
        core_start = core_end

    return np.concatenate(wav_chunks, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX TTS: Hojo-TTS-Light-llm.onnx (KV-cache) + Hojo codec ONNX."
    )
    parser.add_argument(
        "--onnx_dir",
        type=str,
        default=DEFAULT_ONNX_DIR,
        help="ONNX directory (LM + tokenizer + codec encoder/decoder).",
    )
    parser.add_argument("--prompt-speech", type=str, required=True, help="Reference speech wav path.")
    parser.add_argument("--text", type=str, required=True, help="Target text to synthesize.")
    parser.add_argument("--prompt-text", type=str, default="", help="Optional transcript of prompt speech.")
    parser.add_argument(
        "--output-wav",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "out.wav"),
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help=">0 enables sampling; 0 or negative uses greedy argmax.",
    )
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--provider",
        type=str,
        default="CUDAExecutionProvider",
        help="ONNX Runtime provider for LM (CUDAExecutionProvider or CPUExecutionProvider).",
    )
    parser.add_argument(
        "--codec_provider",
        type=str,
        default="",
        help="ORT provider for codec (default: same as --provider).",
    )
    parser.add_argument(
        "--use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable KV-cache decoding for ORTModelForCausalLM (default: True).",
    )
    parser.add_argument(
        "--use_io_binding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="ORT IO binding for LM (default: on for CUDA, off for CPU).",
    )
    parser.add_argument(
        "--strict_cuda",
        action="store_true",
        help="When using CUDA, do not fall back to CPUExecutionProvider.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.output_wav)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    onnx_dir = os.path.abspath(args.onnx_dir)
    llm_onnx = os.path.join(onnx_dir, _LLM_ONNX_NAME)
    enc_onnx = os.path.join(onnx_dir, "Hojo-TTS-Light-encoder.onnx")
    dec_onnx = os.path.join(onnx_dir, "Hojo-TTS-Light-decoder.onnx")

    _validate_onnx_dir(onnx_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available = ort.get_available_providers()
    if args.provider == "CUDAExecutionProvider":
        if device.type != "cuda":
            if args.strict_cuda:
                raise RuntimeError("--provider CUDAExecutionProvider but torch.cuda is unavailable.")
            print("[WARN] CUDA unavailable; falling back to CPUExecutionProvider for LM.")
            lm_providers = ["CPUExecutionProvider"]
        elif "CUDAExecutionProvider" not in available:
            if args.strict_cuda:
                raise RuntimeError(f"CUDAExecutionProvider unavailable: {available}")
            print("[WARN] CUDAExecutionProvider unavailable; using CPUExecutionProvider for LM.")
            lm_providers = ["CPUExecutionProvider"]
        else:
            lm_providers = ["CUDAExecutionProvider"] if args.strict_cuda else [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
    else:
        lm_providers = ["CPUExecutionProvider"]

    codec_provider = args.codec_provider or lm_providers[0]
    codec_providers = [codec_provider]
    if codec_provider == "CUDAExecutionProvider" and not args.strict_cuda:
        codec_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    use_io_binding = args.use_io_binding
    if use_io_binding is None:
        use_io_binding = lm_providers[0] == "CUDAExecutionProvider"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    llm_file_name = _LLM_ONNX_NAME
    print(f"[INFO] Loading ORT LM from: {llm_onnx}")
    print(f"[INFO] Tokenizer: {onnx_dir}")
    print(
        f"[INFO] ORT LM options: provider={lm_providers[0]} "
        f"use_cache={args.use_cache} use_io_binding={use_io_binding}"
    )
    ort_model = ORTModelForCausalLM.from_pretrained(
        onnx_dir,
        file_name=llm_file_name,
        use_cache=args.use_cache,
        use_io_binding=use_io_binding,
        provider=lm_providers[0],
        providers=lm_providers,
    )
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(onnx_dir)

    print(f"[INFO] Loading codec encoder={enc_onnx}")
    print(f"[INFO] Loading codec decoder={dec_onnx} provider={codec_providers[0]}")
    sess_enc = ort.InferenceSession(enc_onnx, providers=codec_providers)
    sess_dec = ort.InferenceSession(dec_onnx, providers=codec_providers)
    enc_np_dtype = _onnx_input_type_to_numpy_dtype(sess_enc.get_inputs()[0].type)

    enc_output = sess_enc.get_outputs()[0]
    enc_input_shape = sess_enc.get_inputs()[0].shape
    enc_fixed_time = enc_input_shape[2] if isinstance(enc_input_shape[2], int) else None

    dec_input_shape = sess_dec.get_inputs()[0].shape
    fixed_chunk_tokens = dec_input_shape[2] if isinstance(dec_input_shape[2], int) else None
    dec_input = sess_dec.get_inputs()[0]
    if not (("int64" in dec_input.type) or (dec_input.name == "vq_codes")):
        raise RuntimeError("decoder ONNX must accept int64 vq_codes.")

    wav, _ = _load_wav(args.prompt_speech, target_sample_rate=CODEC_SAMPLE_RATE)
    wav_np = wav.unsqueeze(0).unsqueeze(0).cpu().numpy().astype(enc_np_dtype, copy=False)
    wav_np = _fit_wav_to_onnx_encoder_time(wav_np, enc_fixed_time)

    enc_out = sess_enc.run(None, {"wav": wav_np})[0]
    encoder_outputs_codes = ("int64" in enc_output.type) or (enc_output.name == "vq_codes")
    if not encoder_outputs_codes:
        raise RuntimeError("encoder ONNX must output int64 vq_codes.")
    prompt_audio_codes = torch.from_numpy(enc_out).to(device=device, dtype=torch.long).view(-1)

    prompt_audio_tokens_str = "".join(
        SPEECH_TOKEN_PATTERN.format(int(cid))
        for cid in prompt_audio_codes.detach().cpu().tolist()
    )
    prompt, target_speech_end_token = build_prompt(
        tokenizer=tokenizer,
        ref_text=args.prompt_text,
        target_text=args.text,
        prompt_audio_tokens_str=prompt_audio_tokens_str,
    )
    input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
    speech_end_id = tokenizer.convert_tokens_to_ids(target_speech_end_token)
    if speech_end_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer cannot find token: {target_speech_end_token}")

    id_to_code = _build_audio_token_id_to_code_table(tokenizer, device)

    gen_input_len = int(input_ids.shape[1])
    t_gen_start = time.perf_counter()
    generated = ort_model.generate(
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        eos_token_id=speech_end_id,
        do_sample=args.temperature > 0.0,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    gen_elapsed_s = time.perf_counter() - t_gen_start
    new_lm_tokens = max(int(generated.shape[1]) - gen_input_len, 0)
    print(
        f"[INFO] ORTModelForCausalLM.generate: input_tokens={gen_input_len} "
        f"new_tokens={new_lm_tokens} elapsed_s={gen_elapsed_s:.4f} "
        f"tok/s={new_lm_tokens / gen_elapsed_s if gen_elapsed_s > 0 else float('nan'):.2f}"
    )

    new_ids = generated[0, gen_input_len:].to(device=device, dtype=torch.long)
    hit_eos = bool((new_ids == speech_end_id).any().item())
    generated_audio_codes = _extract_audio_codes_from_generated_tail(
        new_ids, speech_end_id=speech_end_id, id_to_code=id_to_code
    )
    if generated_audio_codes.numel() == 0:
        raise RuntimeError("No generated audio codes were produced.")

    wav_out = decode_codes_with_onnx_decoder(
        sess_dec=sess_dec,
        codes=generated_audio_codes.contiguous(),
        device=device,
        fixed_chunk_tokens=fixed_chunk_tokens,
    )
    wav_out = wav_out.astype(np.float32, copy=False)
    sf.write(args.output_wav, wav_out, 24000)

    wav_dur_s = len(wav_out) / 24000
    print(
        f"[INFO] Audio decode: codes={int(generated_audio_codes.numel())} "
        f"duration_s={wav_dur_s:.2f} hit_eos={hit_eos}"
    )
    if not hit_eos and new_lm_tokens >= args.max_new_tokens:
        print(
            f"[WARN] Generation stopped at max_new_tokens={args.max_new_tokens} "
            f"without {target_speech_end_token}; output may be truncated."
        )
    print("[OK] output_wav:", args.output_wav)
    print("[NOTE] lm_providers:", lm_providers)
    print("[NOTE] codec_providers:", codec_providers)


if __name__ == "__main__":
    main()
