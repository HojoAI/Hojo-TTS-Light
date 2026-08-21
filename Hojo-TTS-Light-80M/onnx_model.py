"""Hojo-TTS-Light ONNX runtime."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from librosa.filters import mel as librosa_mel_fn

RELEASE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = RELEASE_ROOT / "models"

LM_ONNX_NAME = "Hojo-TTS-Light-llm.onnx"
CODEC_ENCODER_NAME = "Hojo-TTS-Light-encoder.onnx"
DECODER_ONNX_NAME = "Hojo-TTS-Light-decoder.onnx"
SPEAKER_ONNX_NAME = "Hojo-TTS-Light-speaker.onnx"
VOICES_NPZ_NAME = "Hojo-TTS-Light-voice.npz"

OUTPUT_SAMPLE_RATE = 24000
CODEC_SAMPLE_RATE = 16000
DEFAULT_TEMPERATURE = 0.8

# Speaker preprocessing 
SPEAKER_MEL_DIM = 128
SPEAKER_EMB_DIM = 2048
SPEAKER_EMB_SECONDS = 6.0
SPEAKER_MEL_N_FFT = 1024
SPEAKER_MEL_HOP_SIZE = 256
SPEAKER_MEL_WIN_SIZE = 1024
SPEAKER_MEL_FMIN = 0
SPEAKER_MEL_FMAX = 12000

# Decoder ISTFT / token 
CODEC_HOP_LENGTH = 480
CODEC_N_FFT = 1920
CODEC_SAMPLES_PER_TOKEN = CODEC_HOP_LENGTH

DEFAULT_SPEAKER_SECONDS = SPEAKER_EMB_SECONDS

REF_TEXT_START_TOKEN = "[ref_text_start]"
REF_TEXT_END_TOKEN = "[ref_text_end]"
TARGET_TEXT_START_TOKEN = "[target_text_start]"
TARGET_TEXT_END_TOKEN = "[target_text_end]"
REF_SPEECH_START_TOKEN = "[ref_speech_start]"
REF_SPEECH_END_TOKEN = "[ref_speech_end]"
TARGET_SPEECH_START_TOKEN = "[target_speech_start]"
TARGET_SPEECH_END_TOKEN = "[target_speech_end]"
SPEECH_START_TOKEN = "[speech_start]"
SPEECH_END_TOKEN = "[speech_end]"
SPEECH_TOKEN_PATTERN = "[{}]"

_AUDIO_TOKEN_REGEX = re.compile(r"^\[(\d+)\]$")
_CUDA_RUNTIME_CONFIGURED = False


def configure_cpu_threads(num_threads: int) -> None:
    if num_threads <= 0:
        return
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = str(num_threads)


def prepare_cuda_device(physical_device_id: int = 0) -> int:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device_id)
        return 0
    return physical_device_id


def configure_cuda_runtime() -> None:
    global _CUDA_RUNTIME_CONFIGURED
    if _CUDA_RUNTIME_CONFIGURED:
        return
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload(cuda=True, cudnn=True)
    _CUDA_RUNTIME_CONFIGURED = True


def _onnx_contains_bfloat16(model) -> bool:
    from onnx import AttributeProto, TensorProto

    bf16 = TensorProto.BFLOAT16

    def _graph_has_bf16(graph) -> bool:
        for init in graph.initializer:
            if init.data_type == bf16:
                return True
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            if vi.type.tensor_type.elem_type == bf16:
                return True
        for node in graph.node:
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == bf16:
                        return True
            for attr in node.attribute:
                if attr.type == AttributeProto.TENSOR and attr.t.data_type == bf16:
                    return True
                if attr.type == AttributeProto.TENSORS and any(t.data_type == bf16 for t in attr.tensors):
                    return True
                if attr.type == AttributeProto.GRAPH and _graph_has_bf16(attr.g):
                    return True
                if attr.type == AttributeProto.GRAPHS and any(_graph_has_bf16(g) for g in attr.graphs):
                    return True
        return False

    return _graph_has_bf16(model.graph)


def _tensor_proto_bf16_to_fp32(tensor) -> None:
    from onnx import TensorProto, numpy_helper

    if tensor.data_type != TensorProto.BFLOAT16:
        return
    arr = numpy_helper.to_array(tensor)
    tensor.CopyFrom(numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=tensor.name))


def _promote_graph_bf16_to_fp32(graph) -> None:
    from onnx import AttributeProto, TensorProto

    bf16, fp32 = TensorProto.BFLOAT16, TensorProto.FLOAT

    for init in graph.initializer:
        _tensor_proto_bf16_to_fp32(init)

    def _fix_type(type_proto) -> None:
        if type_proto.HasField("tensor_type") and type_proto.tensor_type.elem_type == bf16:
            type_proto.tensor_type.elem_type = fp32

    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        _fix_type(vi.type)

    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == bf16:
                    attr.i = fp32
        for attr in node.attribute:
            if attr.type == AttributeProto.GRAPH:
                _promote_graph_bf16_to_fp32(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for sub_graph in attr.graphs:
                    _promote_graph_bf16_to_fp32(sub_graph)
            elif attr.type == AttributeProto.TENSOR:
                _tensor_proto_bf16_to_fp32(attr.t)
            elif attr.type == AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    _tensor_proto_bf16_to_fp32(tensor)
    del graph.value_info[:]


def _promote_bf16_onnx_to_fp32(model):
    _promote_graph_bf16_to_fp32(model.graph)
    return model


def _ort_model_source(model_path: str, *, promote_bf16_for_cpu: bool):
    if not promote_bf16_for_cpu:
        return model_path
    try:
        import onnx
    except ImportError:
        return model_path
    model = onnx.load(model_path, load_external_data=True)
    if not _onnx_contains_bfloat16(model):
        return model_path
    _promote_bf16_onnx_to_fp32(model)
    return model.SerializeToString()


def _build_ort_session(
    model_path: str,
    provider: str,
    *,
    device_id: int = 0,
    num_threads: int = 0,
) -> ort.InferenceSession:
    providers = (
        [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        if provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if num_threads > 0:
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = num_threads
    promote_bf16 = provider != "cuda"
    return ort.InferenceSession(
        _ort_model_source(model_path, promote_bf16_for_cpu=promote_bf16),
        sess_options=so,
        providers=providers,
    )


class _PromptTokenizer:
    def __init__(self, resources_dir: str) -> None:
        from tokenizers import Tokenizer

        with open(os.path.join(resources_dir, "tokenizer_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        self._tok = Tokenizer.from_file(os.path.join(resources_dir, "tokenizer.json"))
        unk = cfg.get("unk_token", "<unk>")
        self.unk_token_id = self._tok.token_to_id(unk) or 0

    def __len__(self) -> int:
        return self._tok.get_vocab_size()

    def convert_tokens_to_ids(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        return self.unk_token_id if token_id is None else token_id

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> np.ndarray:
        encoded = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return np.array([encoded.ids], dtype=np.int64)


def _load_token_embedding(models_dir: str) -> np.ndarray:
    npz_path = os.path.join(models_dir, VOICES_NPZ_NAME)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Missing [{npz_path}] ({VOICES_NPZ_NAME}).")
    with np.load(npz_path, allow_pickle=True) as data:
        if "token_embedding" not in data:
            raise ValueError(f"{VOICES_NPZ_NAME} must contain key token_embedding.")
        return np.asarray(data["token_embedding"], dtype=np.float32)


def _resample_mono(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    gcd = math.gcd(src_sr, dst_sr)
    return resample_poly(wav, dst_sr // gcd, src_sr // gcd).astype(np.float32, copy=False)


def _load_prompt_wav(path: str, *, target_sample_rate: int = CODEC_SAMPLE_RATE) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return _resample_mono(np.asarray(data, dtype=np.float32), int(sr), target_sample_rate)


def _build_ref_codec_prompt(
    tokenizer: _PromptTokenizer,
    ref_text: str,
    target_text: str,
    prompt_audio_codes: np.ndarray,
) -> np.ndarray:
    ref_speech_start = REF_SPEECH_START_TOKEN
    ref_speech_end = REF_SPEECH_END_TOKEN
    target_speech_start = TARGET_SPEECH_START_TOKEN
    if tokenizer.convert_tokens_to_ids(target_speech_start) == tokenizer.unk_token_id:
        ref_speech_start = SPEECH_START_TOKEN
        ref_speech_end = SPEECH_END_TOKEN
        target_speech_start = SPEECH_START_TOKEN

    prompt_audio_tokens = "".join(
        SPEECH_TOKEN_PATTERN.format(int(code)) for code in prompt_audio_codes.tolist()
    )
    prompt = (
        f"{REF_TEXT_START_TOKEN}{ref_text}{REF_TEXT_END_TOKEN} "
        f"{TARGET_TEXT_START_TOKEN}{target_text}{TARGET_TEXT_END_TOKEN}"
        f"{ref_speech_start}{prompt_audio_tokens}{ref_speech_end}"
        f"{target_speech_start}"
    )
    return tokenizer.encode(prompt, add_special_tokens=True)


def _build_audio_token_id_to_code_table(tokenizer: _PromptTokenizer) -> np.ndarray:
    table = np.full(len(tokenizer), -1, dtype=np.int64)
    for tid in range(len(tokenizer)):
        token_str = tokenizer.decode([tid], skip_special_tokens=False).strip()
        match = _AUDIO_TOKEN_REGEX.match(token_str)
        if match is not None:
            table[tid] = int(match.group(1))
    return table


def _extract_audio_token_positions(
    generated_ids: np.ndarray,
    prompt_len: int,
    speech_end_id: int,
    id_to_code: np.ndarray,
) -> np.ndarray:
    ends = np.where(generated_ids == speech_end_id)[0]
    seq = generated_ids[: int(ends[0])] if ends.size else generated_ids
    rel = np.flatnonzero(id_to_code[seq] >= 0)
    return (prompt_len + rel).astype(np.int64)


def _empty_lm_past(num_layers: int, *, batch: int = 1) -> dict[str, np.ndarray]:
    past: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        past[f"past_key_values.{layer}.key"] = np.zeros((batch, 1, 0, 128), dtype=np.float32)
        past[f"past_key_values.{layer}.value"] = np.zeros((batch, 1, 0, 128), dtype=np.float32)
    return past


def _sample_next_token(
    logits: np.ndarray,
    *,
    temperature: float,
    top_p: float,
    generated_ids: list[int],
    repetition_penalty: float,
) -> int:
    row = logits.astype(np.float32).copy()
    if repetition_penalty != 1.0 and generated_ids:
        for token_id in set(generated_ids):
            value = row[token_id]
            row[token_id] = value / repetition_penalty if value > 0 else value * repetition_penalty
    if temperature <= 0.0:
        return int(row.argmax())
    row = (row / temperature) - row.max()
    probs = np.exp(row)
    probs /= probs.sum()
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        cutoff = cumulative > top_p
        if cutoff.any():
            keep = order[: int(np.argmax(cutoff)) + 1]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep] = True
            probs = np.where(mask, probs, 0.0)
            probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def _mel_spectrogram(
    waveform: torch.Tensor,
    *,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int,
) -> torch.Tensor:
    device = waveform.device
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(device)
    hann_window = torch.hann_window(win_size).to(device)
    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(torch.matmul(mel_basis, spec), min=1e-5))


def _prepare_speaker_wav(wav: np.ndarray, sr: int, target_sr: int, seconds: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = _resample_mono(wav, sr, target_sr)
    max_samples = int(round(seconds * target_sr))
    if wav.shape[0] > max_samples:
        return wav[:max_samples]
    if wav.shape[0] < max_samples:
        return np.pad(wav, (0, max_samples - wav.shape[0]))
    return wav


class _ISTFT:
    def __init__(self, n_fft: int, hop_length: int, win_length: int) -> None:
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = np.hanning(win_length + 1)[:-1].astype(np.float32)
        self.pad = (win_length - hop_length) // 2

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        outputs = []
        for b in range(spec.shape[0]):
            ifft = np.fft.irfft(spec[b], n=self.n_fft, axis=0, norm="backward")
            ifft *= self.window[:, None]
            y = self._overlap_add(ifft)[self.pad : -self.pad]
            env = self._overlap_add(np.broadcast_to(self.window[:, None] ** 2, ifft.shape))[self.pad : -self.pad]
            outputs.append((y / np.maximum(env, 1e-11)).astype(np.float32))
        return np.stack(outputs, axis=0)

    def _overlap_add(self, frames: np.ndarray) -> np.ndarray:
        win_length, num_frames = frames.shape
        out = np.zeros((num_frames - 1) * self.hop_length + win_length, dtype=frames.dtype)
        for i in range(num_frames):
            start = i * self.hop_length
            out[start : start + win_length] += frames[:, i]
        return out


def _wav_from_mag_phase(mag: np.ndarray, phase: np.ndarray, istft: _ISTFT) -> np.ndarray:
    log_mag = np.log(np.maximum(mag, 1e-12))
    x_pred = np.concatenate([log_mag, phase], axis=0)
    mag_exp, phase_use = np.split(x_pred, 2, axis=0)
    mag_exp = np.exp(mag_exp).clip(max=1e2)
    spec = mag_exp * np.cos(phase_use) + 1j * mag_exp * np.sin(phase_use)
    return istft(spec[np.newaxis, ...])[0]


def _require_unified_lm(session: ort.InferenceSession) -> tuple[int, int, int]:
    inputs = {inp.name for inp in session.get_inputs()}
    outputs = [out.name for out in session.get_outputs()]
    if "inputs_embeds" not in inputs:
        raise ValueError(f"{LM_ONNX_NAME} must be DD-SFT unified LM (inputs_embeds).")
    if "last_hidden_state" not in outputs:
        raise ValueError(f"{LM_ONNX_NAME} must export last_hidden_state.")
    logits_i = outputs.index("logits")
    hidden_i = outputs.index("last_hidden_state")
    past_start = max(logits_i, hidden_i) + 1
    return logits_i, hidden_i, past_start


def _require_merged_decoder(session: ort.InferenceSession) -> None:
    required = {"hidden_states", "coarse_embeddings", "speaker_embedding", "valid_mask"}
    missing = required - {inp.name for inp in session.get_inputs()}
    if missing:
        raise ValueError(
            f"{DECODER_ONNX_NAME} must be merged DD-SFT graph; missing inputs: {sorted(missing)}"
        )


class HojoTTSLightOnnx:
    """Hojo-TTS-Light ONNX runtime."""

    def __init__(
        self,
        models_dir: str | Path | None = None,
        *,
        provider: str = "cpu",
        device_id: int = 0,
        num_threads: int = 0,
    ) -> None:
        self.models_dir = os.path.abspath(str(models_dir or DEFAULT_MODELS_DIR))
        if provider == "cuda":
            configure_cuda_runtime()
            device_id = prepare_cuda_device(device_id)

        paths = {
            LM_ONNX_NAME: os.path.join(self.models_dir, LM_ONNX_NAME),
            CODEC_ENCODER_NAME: os.path.join(self.models_dir, CODEC_ENCODER_NAME),
            DECODER_ONNX_NAME: os.path.join(self.models_dir, DECODER_ONNX_NAME),
            SPEAKER_ONNX_NAME: os.path.join(self.models_dir, SPEAKER_ONNX_NAME),
            VOICES_NPZ_NAME: os.path.join(self.models_dir, VOICES_NPZ_NAME),
        }
        for label, path in paths.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing [{path}] ({label}).")

        self.lm = _build_ort_session(paths[LM_ONNX_NAME], provider, device_id=device_id, num_threads=num_threads)
        self.codec_encoder = _build_ort_session(
            paths[CODEC_ENCODER_NAME],
            "cuda" if provider == "cuda" else "cpu",
            device_id=device_id,
            num_threads=num_threads,
        )
        self.decoder = _build_ort_session(
            paths[DECODER_ONNX_NAME], provider, device_id=device_id, num_threads=num_threads
        )
        self.speaker_encoder = _build_ort_session(
            paths[SPEAKER_ONNX_NAME], provider, device_id=device_id, num_threads=num_threads
        )

        self._lm_logits_index, self._lm_hidden_index, self._lm_past_start = _require_unified_lm(self.lm)
        _require_merged_decoder(self.decoder)

        self.token_embedding = _load_token_embedding(self.models_dir)
        self.tokenizer = _PromptTokenizer(self.models_dir)
        self.id_to_code = _build_audio_token_id_to_code_table(self.tokenizer)
        self.speech_end_id = self.tokenizer.convert_tokens_to_ids(TARGET_SPEECH_END_TOKEN)
        if self.speech_end_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer missing {TARGET_SPEECH_END_TOKEN!r}")

        with open(os.path.join(self.models_dir, "config.json"), encoding="utf-8") as f:
            self.num_layers = int(json.load(f)["num_hidden_layers"])

        self.istft = _ISTFT(
            n_fft=CODEC_N_FFT,
            hop_length=CODEC_HOP_LENGTH,
            win_length=CODEC_N_FFT,
        )

        enc_in = self.codec_encoder.get_inputs()[0]
        self.enc_input_name = enc_in.name
        self.enc_np_dtype = np.float16 if "float16" in enc_in.type else np.float32

    @property
    def sample_rate(self) -> int:
        return OUTPUT_SAMPLE_RATE

    def _encode_ref_codes(self, prompt_wav_path: str) -> np.ndarray:
        wav = _load_prompt_wav(prompt_wav_path).reshape(1, 1, -1).astype(self.enc_np_dtype)
        return self.codec_encoder.run(None, {self.enc_input_name: wav})[0].reshape(-1).astype(np.int64)

    def _encode_speaker(self, ref_wav_path: str) -> np.ndarray:
        wav, sr = sf.read(ref_wav_path, dtype="float32", always_2d=False)
        wav = _prepare_speaker_wav(wav, int(sr), OUTPUT_SAMPLE_RATE, SPEAKER_EMB_SECONDS)
        mels = _mel_spectrogram(
            torch.from_numpy(wav).unsqueeze(0),
            n_fft=SPEAKER_MEL_N_FFT,
            num_mels=SPEAKER_MEL_DIM,
            sampling_rate=OUTPUT_SAMPLE_RATE,
            hop_size=SPEAKER_MEL_HOP_SIZE,
            win_size=SPEAKER_MEL_WIN_SIZE,
            fmin=SPEAKER_MEL_FMIN,
            fmax=SPEAKER_MEL_FMAX,
        ).transpose(1, 2)
        mel_in = self.speaker_encoder.get_inputs()[0].name
        return self.speaker_encoder.run(None, {mel_in: mels.numpy().astype(np.float32)})[0].astype(np.float32)

    def _generate_coarse_tokens(
        self,
        input_ids: np.ndarray,
        *,
        max_new_tokens: int,
        min_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_len = int(input_ids.shape[1])
        prefill_out = self.lm.run(
            None,
            {
                "inputs_embeds": self.token_embedding[input_ids].astype(np.float32),
                "position_ids": np.arange(seq_len, dtype=np.int64)[None, :],
                **_empty_lm_past(self.num_layers),
            },
        )
        logits = prefill_out[self._lm_logits_index]
        hidden = prefill_out[self._lm_hidden_index]
        past = prefill_out[self._lm_past_start :]
        hidden_chunks = [hidden]

        generated_ids: list[int] = []
        next_token = _sample_next_token(
            logits[0, -1],
            temperature=temperature,
            top_p=top_p,
            generated_ids=generated_ids,
            repetition_penalty=repetition_penalty,
        )
        generated_ids.append(next_token)
        cur_len = seq_len

        for _ in range(max_new_tokens - 1):
            if len(generated_ids) >= min_new_tokens and next_token == self.speech_end_id:
                break
            feed = {
                "inputs_embeds": self.token_embedding[np.array([[next_token]], dtype=np.int64)].astype(np.float32),
                "position_ids": np.array([[cur_len]], dtype=np.int64),
            }
            for layer in range(self.num_layers):
                feed[f"past_key_values.{layer}.key"] = past[layer * 2]
                feed[f"past_key_values.{layer}.value"] = past[layer * 2 + 1]
            decode_out = self.lm.run(None, feed)
            logits = decode_out[self._lm_logits_index]
            hidden = decode_out[self._lm_hidden_index]
            past = decode_out[self._lm_past_start :]
            hidden_chunks.append(hidden)
            next_token = _sample_next_token(
                logits[0, -1],
                temperature=temperature,
                top_p=top_p,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)
            cur_len += 1

        return np.array(generated_ids, dtype=np.int64), np.concatenate(hidden_chunks, axis=1)

    def _decode_from_coarse(
        self,
        prompt_ids: np.ndarray,
        generated: np.ndarray,
        last_hidden: np.ndarray,
        speaker_vec: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        prompt_len = int(prompt_ids.shape[1])
        full_ids = np.concatenate([prompt_ids[0], generated], axis=0)[None, :].astype(np.int64)
        audio_positions = _extract_audio_token_positions(
            generated, prompt_len, self.speech_end_id, self.id_to_code
        )
        if audio_positions.size == 0:
            raise RuntimeError("LM did not generate valid audio tokens like [123].")

        return self.decoder.run(
            None,
            {
                "hidden_states": last_hidden[:, audio_positions - 1, :].astype(np.float32),
                "coarse_embeddings": self.token_embedding[full_ids[:, audio_positions]].astype(np.float32),
                "speaker_embedding": speaker_vec.reshape(1, -1).astype(np.float32),
                "valid_mask": np.ones((1, int(audio_positions.size)), dtype=bool),
            },
        )

    def generate(
        self,
        text: str,
        *,
        prompt_speech: str,
        prompt_text: str = "",
        max_new_tokens: int = 2048,
        min_new_tokens: int = 10,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        seed: int = 42,
    ) -> np.ndarray:
        np.random.seed(seed)
        prompt_codes = self._encode_ref_codes(prompt_speech)
        input_ids = _build_ref_codec_prompt(self.tokenizer, prompt_text, text, prompt_codes)
        speaker_vec = self._encode_speaker(prompt_speech)
        generated, last_hidden = self._generate_coarse_tokens(
            input_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        mag, phase = self._decode_from_coarse(input_ids, generated, last_hidden, speaker_vec)
        return _wav_from_mag_phase(mag, phase, self.istft)
