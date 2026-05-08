#!/usr/bin/env python3
import argparse
import glob
import os

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
from transformers import AutoTokenizer


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


def _resolve_generation_tokens(tokenizer):
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


def _build_audio_token_id_to_code_table(tokenizer, device: torch.device) -> dict[int, torch.Tensor]:
    id_to_code: dict[int, torch.Tensor] = {}
    token_zero = SPEECH_TOKEN_PATTERN.format(0)
    token_one = SPEECH_TOKEN_PATTERN.format(1)
    left = 0
    while left < min(len(token_zero), len(token_one)) and token_zero[left] == token_one[left]:
        left += 1
    right = 0
    while (
        right < (len(token_zero) - left)
        and right < (len(token_one) - left)
        and token_zero[len(token_zero) - 1 - right] == token_one[len(token_one) - 1 - right]
    ):
        right += 1
    prefix = token_zero[:left]
    suffix = token_zero[len(token_zero) - right :] if right > 0 else ""

    for tid in range(len(tokenizer)):
        token_str = tokenizer.decode([tid], skip_special_tokens=False).strip()
        if not token_str.startswith(prefix):
            continue
        if suffix and (not token_str.endswith(suffix)):
            continue
        middle = token_str[len(prefix) : len(token_str) - len(suffix) if suffix else None]
        if not middle:
            continue
        if middle[0] == "-":
            if not middle[1:].isdigit():
                continue
        elif not middle.isdigit():
            continue
        id_to_code[tid] = torch.tensor(int(middle), dtype=torch.long, device=device)
    return id_to_code


def _extract_audio_codes_from_generated_tail(
    generated_ids: torch.Tensor,
    speech_end_id: int,
    id_to_code: dict[int, torch.Tensor],
) -> torch.Tensor:
    if generated_ids.dim() != 1:
        generated_ids = generated_ids.view(-1)
    eos_pos = (generated_ids == speech_end_id).nonzero(as_tuple=False)
    if eos_pos.numel() > 0:
        generated_ids = generated_ids[: int(eos_pos[0].item())]

    codes = [id_to_code[int(tid.item())] for tid in generated_ids if int(tid.item()) in id_to_code]
    if not codes:
        return torch.empty((0,), dtype=torch.long, device=generated_ids.device)
    return torch.stack(codes, dim=0).to(dtype=torch.long)


def _onnx_input_type_to_numpy_dtype(type_str: str) -> np.dtype:
    if type_str != "tensor(float16)":
        raise ValueError(
            f"only fp16 encoder input is supported, but got ONNX input type: {type_str}"
        )
    return np.float16


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


def onnx_ar_generate_new_ids(
    sess_llm: ort.InferenceSession,
    input_ids: torch.Tensor,
    eos_token_id: int,
    max_new_tokens: int,
    min_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    cur = input_ids[0].detach().cpu().numpy().astype(np.int64)
    generated = []
    for step in range(max_new_tokens):
        logits = sess_llm.run(None, {"input_ids": cur.reshape(1, -1)})[0]
        next_id = int(np.argmax(logits[0, -1, :]))
        generated.append(next_id)
        cur = np.concatenate([cur, np.array([next_id], dtype=np.int64)], axis=0)
        if next_id == eos_token_id and step + 1 >= min_new_tokens:
            break
    return torch.tensor(generated, dtype=torch.long, device=device)


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
            "This script only supports new ONNX exports where decoder input is int64 vq_codes."
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
    parser = argparse.ArgumentParser(description="Single-sample ONNX inference (prompt speech + text).")
    parser.add_argument("--onnx_dir", type=str, required=True)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/speech/users/tts_common/tts_pingce/tts_envs/tts_wfp_v1/model_for_test_pinyin",
    )
    parser.add_argument("--prompt-speech", type=str, required=True, help="Reference speech wav path.")
    parser.add_argument("--text", type=str, required=True, help="Target text to synthesize.")
    parser.add_argument("--prompt-text", type=str, default="", help="Optional transcript of prompt speech.")
    parser.add_argument(
        "--output-wav",
        type=str,
        default="/speech/users/tts_common/tts_pingce/tts_envs/tts_wfp_v1/test_result/onnx_single.wav",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--llm_provider", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--codec_provider", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--strict_cuda",
        action="store_true",
        help="When --device cuda, use CUDAExecutionProvider only (no CPU fallback).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.dirname(args.output_wav)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    llm_onnx = os.path.join(args.onnx_dir, "Hojo-TTS-Light-llm.onnx")
    enc_onnx = os.path.join(args.onnx_dir, "Hojo-TTS-Light-encoder.onnx")
    dec_onnx = os.path.join(args.onnx_dir, "Hojo-TTS-Light-decoder.onnx")
    for p in (llm_onnx, enc_onnx, dec_onnx):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"onnx file missing: {p}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was set but torch.cuda is unavailable.")
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "--device cuda was set but CUDAExecutionProvider is unavailable. "
                f"available providers: {available}"
            )
        providers = ["CUDAExecutionProvider"] if args.strict_cuda else [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        device = torch.device("cuda")
    else:
        providers = ["CPUExecutionProvider"]
        device = torch.device("cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.isdir(args.tokenizer_path):
        raise FileNotFoundError(f"tokenizer_path is not a directory: {args.tokenizer_path}")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(args.tokenizer_path, name)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing tokenizer resource file: {p}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    def _providers_for(kind: str) -> list[str]:
        choice = getattr(args, f"{kind}_provider")
        if choice == "auto":
            return providers
        if choice == "cpu":
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    llm_providers = _providers_for("llm")
    codec_providers = _providers_for("codec")
    sess_llm = ort.InferenceSession(llm_onnx, providers=llm_providers)
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
        raise RuntimeError("decoder ONNX must accept int64 vq_codes; old exports are not supported.")

    wav, _ = _load_wav(args.prompt_speech, target_sample_rate=CODEC_SAMPLE_RATE)
    prompt_wav = wav.to(device)
    # Encoder expects [B, 1, T].
    wav_np = prompt_wav.unsqueeze(0).unsqueeze(0).cpu().numpy().astype(enc_np_dtype, copy=False)
    wav_np = _fit_wav_to_onnx_encoder_time(wav_np, enc_fixed_time)

    enc_out = sess_enc.run(None, {"wav": wav_np})[0]
    encoder_outputs_codes = ("int64" in enc_output.type) or (enc_output.name == "vq_codes")
    if not encoder_outputs_codes:
        raise RuntimeError("encoder ONNX must output int64 vq_codes; old exports are not supported.")
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
    input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(device)
    speech_end_id = tokenizer.convert_tokens_to_ids(target_speech_end_token)

    id_to_code = _build_audio_token_id_to_code_table(tokenizer, device)
    new_ids = onnx_ar_generate_new_ids(
        sess_llm=sess_llm,
        input_ids=input_ids,
        eos_token_id=speech_end_id,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        device=device,
    )
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
    print("[OK] output_wav:", args.output_wav)
    print("[NOTE] llm_providers:", llm_providers)
    print("[NOTE] codec_providers:", codec_providers)


if __name__ == "__main__":
    main()
