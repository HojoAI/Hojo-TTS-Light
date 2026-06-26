"""ONNX runtime for Hojo TTS Light (model load + inference)."""

from __future__ import annotations

import json
import json as _json
import os
import os as _os
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort

RELEASE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = RELEASE_ROOT / "models"
LM_ONNX_NAME = "Hojo-TTS-Light-40M-llm.onnx"
CODEC_ONNX_NAME = "Hojo-TTS-Light-40M-decoder.onnx"
VOICES_NPZ_NAME = "Hojo-TTS-Light-40M-voice.npz"
DEFAULT_VOICES = DEFAULT_MODELS_DIR / VOICES_NPZ_NAME
OUTPUT_SAMPLE_RATE = 24000

TARGET_TEXT_START_TOKEN = "[target_text_start]"
TARGET_TEXT_END_TOKEN = "[target_text_end]"
SPK_START_TOKEN = "[spk_start]"
SPK_END_TOKEN = "[spk_end]"
TARGET_SPEECH_START_TOKEN = "[target_speech_start]"
TARGET_SPEECH_END_TOKEN = "[target_speech_end]"
NUM_SPEAKER_INJECTION_SLOTS = 16
SPEAKER_PLACEHOLDER_TOKENS = tuple(
    f"[spk_emb_{i}]" for i in range(NUM_SPEAKER_INJECTION_SLOTS)
)
_AUDIO_TOKEN_REGEX = re.compile(r"^\[(\d+)\]$")
DEFAULT_TEMPERATURE = 0.8


class _PromptTokenizer:
    """Thin wrapper around ``tokenizers`` (no transformers dependency)."""

    def __init__(self, resources_dir: str) -> None:
        from tokenizers import Tokenizer

        with open(
            _os.path.join(resources_dir, "tokenizer_config.json"), encoding="utf-8"
        ) as f:
            tokenizer_config = _json.load(f)

        self._tok = Tokenizer.from_file(_os.path.join(resources_dir, "tokenizer.json"))
        unk = tokenizer_config.get("unk_token", "<unk>")
        self.unk_token_id = self._tok.token_to_id(unk)
        if self.unk_token_id is None:
            self.unk_token_id = 0

    def __len__(self) -> int:
        return self._tok.get_vocab_size()

    def convert_tokens_to_ids(self, token: str) -> int:
        token_id = self._tok.token_to_id(token)
        return self.unk_token_id if token_id is None else token_id

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self, text: str, *, add_special_tokens: bool = True, return_tensors: str = "np"
    ) -> dict[str, np.ndarray]:
        if return_tensors != "np":
            raise ValueError("Only return_tensors='np' is supported.")
        encoded = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": np.array([encoded.ids], dtype=np.int64)}


def configure_cpu_threads(num_threads: int) -> None:
    """Limit ORT / BLAS CPU parallelism (call before creating sessions)."""
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


def _build_ort_session(
    model_path: str, provider: str, device_id: int = 0, num_threads: int = 0
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
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


def build_speaker_prompt(text: str) -> str:
    placeholders = "".join(SPEAKER_PLACEHOLDER_TOKENS)
    return (
        f"{TARGET_TEXT_START_TOKEN}{text}{TARGET_TEXT_END_TOKEN}"
        f"{SPK_START_TOKEN}{placeholders}{SPK_END_TOKEN}"
        f"{TARGET_SPEECH_START_TOKEN}"
    )


class VoiceBank:
    """Precomputed speaker embeddings from Hojo-TTS-Light-40M-voice.npz (opaque voice IDs only)."""

    def __init__(self, voices_npz: str | Path) -> None:
        voices_npz = os.path.abspath(str(voices_npz))
        if not os.path.isfile(voices_npz):
            raise FileNotFoundError(f"Missing voices file: {voices_npz}")
        data = np.load(voices_npz, allow_pickle=True)
        if "voice_ids" not in data:
            raise ValueError(
                f"{voices_npz} must contain voice_ids; regenerate with export_speaker_onnx."
            )
        self.voice_ids = [str(v) for v in data["voice_ids"]]
        self.speaker_embeds = np.asarray(data["speaker_embeds"], dtype=np.float32)
        self._id_to_idx = {voice_id: idx for idx, voice_id in enumerate(self.voice_ids)}

    def list_voices(self) -> list[str]:
        return sorted(self.voice_ids)

    def get_speaker_embeds(self, voice: str) -> np.ndarray:
        voice_id = str(voice)
        if voice_id not in self._id_to_idx:
            raise KeyError(f"Unknown voice {voice_id!r}")
        idx = self._id_to_idx[voice_id]
        return self.speaker_embeds[idx : idx + 1].astype(np.float16, copy=False)


def _build_audio_token_id_to_code_table(tokenizer) -> np.ndarray:
    table = np.full((len(tokenizer),), -1, dtype=np.int64)
    for tid in range(len(tokenizer)):
        token_str = tokenizer.decode([tid], skip_special_tokens=False).strip()
        match = _AUDIO_TOKEN_REGEX.match(token_str)
        if match is not None:
            table[tid] = int(match.group(1))
    return table


def extract_audio_codes(
    generated_ids: np.ndarray, speech_end_id: int, id_to_code: np.ndarray
) -> np.ndarray:
    ends = np.where(generated_ids == speech_end_id)[0]
    seq = generated_ids[: int(ends[0])] if ends.size else generated_ids
    codes = id_to_code[seq]
    return codes[codes >= 0]


def sample_next_token(
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
    row = row / temperature
    row = row - row.max()
    probs = np.exp(row)
    probs = probs / probs.sum()
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        cutoff = cumulative > top_p
        if cutoff.any():
            cutoff_idx = int(np.argmax(cutoff))
            keep = order[: cutoff_idx + 1]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep] = True
            probs = np.where(mask, probs, 0.0)
            probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def _hann_window(win_length: int) -> np.ndarray:
    return np.hanning(win_length + 1)[:-1].astype(np.float32)


def _overlap_add(frames: np.ndarray, hop_length: int) -> np.ndarray:
    win_length, num_frames = frames.shape
    output_size = (num_frames - 1) * hop_length + win_length
    out = np.zeros(output_size, dtype=frames.dtype)
    for i in range(num_frames):
        start = i * hop_length
        out[start : start + win_length] += frames[:, i]
    return out


class ISTFT:
    """Minimal ISTFT for codec mag/phase -> waveform (NumPy only)."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        if padding not in ("center", "same"):
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = _hann_window(win_length)

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        return self.forward(spec)

    def forward(self, spec: np.ndarray) -> np.ndarray:
        if self.padding != "same":
            raise NotImplementedError("Only padding='same' is supported.")

        spec = np.asarray(spec)
        if spec.ndim != 3:
            raise ValueError("Expected a 3D spectrogram array (batch, freq, time).")

        batch, _freq_bins, t = spec.shape
        pad = (self.win_length - self.hop_length) // 2
        outputs: list[np.ndarray] = []

        for b in range(batch):
            ifft = np.fft.irfft(spec[b], n=self.n_fft, axis=0, norm="backward")
            ifft = ifft * self.window[:, None]

            y = _overlap_add(ifft, self.hop_length)[pad:-pad]

            window_sq_frames = np.broadcast_to(self.window[:, None] ** 2, (self.win_length, t))
            window_envelope = _overlap_add(window_sq_frames, self.hop_length)[pad:-pad]

            if not np.all(window_envelope > 1e-11):
                raise AssertionError("window envelope has near-zero values")
            outputs.append((y / window_envelope).astype(np.float32, copy=False))

        return np.stack(outputs, axis=0)


def wav_from_mag_phase(mag: np.ndarray, phase: np.ndarray, istft_module) -> np.ndarray:
    mag = np.asarray(mag, dtype=np.float32)
    phase = np.asarray(phase, dtype=np.float32)
    log_mag = np.log(np.maximum(mag, 1e-12))
    x_pred = np.concatenate([log_mag, phase], axis=0)
    mag_exp, phase_use = np.split(x_pred, 2, axis=0)
    mag_exp = np.exp(mag_exp).clip(max=1e2)
    spec = mag_exp * np.cos(phase_use) + 1j * mag_exp * np.sin(phase_use)
    wav = istft_module(spec[np.newaxis, ...])[0]
    return wav.astype(np.float32, copy=False)


def empty_lm_past(num_layers: int, *, batch: int = 1) -> dict[str, np.ndarray]:
    past: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        past[f"past_key_values.{layer}.key"] = np.zeros((batch, 1, 0, 128), dtype=np.float16)
        past[f"past_key_values.{layer}.value"] = np.zeros((batch, 1, 0, 128), dtype=np.float16)
    return past


def _session_input_names(session: ort.InferenceSession) -> set[str]:
    return {inp.name for inp in session.get_inputs()}


class HojoTTSLightOnnx:
    """Low-level ONNX TTS runtime (LM + codec + voices)."""

    def __init__(
        self,
        models_dir: str | Path | None = None,
        voices_npz: str | Path | None = None,
        *,
        provider: str = "cpu",
        device_id: int = 0,
        num_threads: int = 0,
    ) -> None:
        self.models_dir = os.path.abspath(str(models_dir or DEFAULT_MODELS_DIR))
        voices_path = voices_npz or os.path.join(self.models_dir, VOICES_NPZ_NAME)

        lm_path = os.path.join(self.models_dir, LM_ONNX_NAME)
        if not os.path.isfile(lm_path):
            raise FileNotFoundError(f"Missing [{lm_path}].")

        self.lm = _build_ort_session(lm_path, provider, device_id, num_threads)
        self._lm_input_names = _session_input_names(self.lm)
        if "is_prefill" not in self._lm_input_names:
            raise ValueError(
                f"{LM_ONNX_NAME} must be a unified LM graph with is_prefill input."
            )

        codec_path = os.path.join(self.models_dir, CODEC_ONNX_NAME)
        if not os.path.isfile(codec_path):
            raise FileNotFoundError(f"Missing [{codec_path}].")
        self.codec_decode = _build_ort_session(codec_path, provider, device_id, num_threads)

        self.voices = VoiceBank(voices_path)
        self.tokenizer = _PromptTokenizer(self.models_dir)
        self.id_to_code = _build_audio_token_id_to_code_table(self.tokenizer)

        self.istft = ISTFT(
            n_fft=480 * 4,
            hop_length=480,
            win_length=480 * 4,
            padding="same",
        )

        with open(os.path.join(self.models_dir, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.num_layers = int(self.config["num_hidden_layers"])

        self.speech_end_id = self.tokenizer.convert_tokens_to_ids(TARGET_SPEECH_END_TOKEN)
        if self.speech_end_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer missing {TARGET_SPEECH_END_TOKEN!r}")

    @property
    def sample_rate(self) -> int:
        return OUTPUT_SAMPLE_RATE

    @property
    def available_voices(self) -> list[str]:
        return self.voices.list_voices()

    def generate(
        self,
        text: str,
        *,
        voice: str,
        max_new_tokens: int = 2048,
        min_new_tokens: int = 10,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        seed: int = 42,
    ) -> np.ndarray:
        """Synthesize speech and return a 1-D float32 waveform @ 24 kHz."""
        np.random.seed(seed)

        prompt = build_speaker_prompt(text)
        input_ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors="np")[
            "input_ids"
        ].astype(np.int64)
        seq_len = int(input_ids.shape[1])

        speaker_embeds = self.voices.get_speaker_embeds(voice)
        attention_mask = np.ones((1, seq_len), dtype=np.int64)
        position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

        prefill_feed = {
            "input_ids": input_ids,
            "speaker_embeds": speaker_embeds.astype(np.float16),
            "position_ids": position_ids,
            "is_prefill": np.array([1], dtype=np.int64),
            **empty_lm_past(self.num_layers),
        }
        if "attention_mask" in self._lm_input_names:
            prefill_feed["attention_mask"] = attention_mask
        prefill_out = self.lm.run(None, prefill_feed)

        logits = prefill_out[0]
        past = prefill_out[1:]

        generated_ids: list[int] = []
        next_token = sample_next_token(
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
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "position_ids": np.array([[cur_len]], dtype=np.int64),
                "speaker_embeds": np.zeros(
                    (1, NUM_SPEAKER_INJECTION_SLOTS, 512), dtype=np.float16
                ),
                "is_prefill": np.array([0], dtype=np.int64),
            }
            if "attention_mask" in self._lm_input_names:
                feed["attention_mask"] = np.ones((1, cur_len + 1), dtype=np.int64)
            for layer in range(self.num_layers):
                feed[f"past_key_values.{layer}.key"] = past[layer * 2]
                feed[f"past_key_values.{layer}.value"] = past[layer * 2 + 1]

            decode_out = self.lm.run(None, feed)
            logits = decode_out[0]
            past = decode_out[1:]
            next_token = sample_next_token(
                logits[0, -1],
                temperature=temperature,
                top_p=top_p,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)
            cur_len += 1

        generated = np.array(generated_ids, dtype=np.int64)
        audio_codes = extract_audio_codes(generated, self.speech_end_id, self.id_to_code)
        if audio_codes.size == 0:
            raise RuntimeError("LM did not generate valid audio tokens like [123].")

        mag, phase = self.codec_decode.run(None, {"codes": audio_codes.astype(np.int64)})
        return wav_from_mag_phase(mag, phase, self.istft)
