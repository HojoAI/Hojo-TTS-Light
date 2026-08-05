"""ONNX runtime for Hojo TTS Light (model load + inference)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort

RELEASE_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = RELEASE_ROOT / "models"
LM_ONNX_NAME = "Hojo-TTS-Light-40M-llm.onnx"
FINE_LOCAL_ONNX_NAME = "Hojo-TTS-Light-40M-fine_local.onnx"
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
            os.path.join(resources_dir, "tokenizer_config.json"), encoding="utf-8"
        ) as f:
            tokenizer_config = json.load(f)

        self._tok = Tokenizer.from_file(os.path.join(resources_dir, "tokenizer.json"))
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


def _onnx_contains_bfloat16(model) -> bool:
    """True if any initializer / tensor type / Cast target / Constant uses BFLOAT16."""
    from onnx import TensorProto, AttributeProto

    bf16 = TensorProto.BFLOAT16
    for init in model.graph.initializer:
        if init.data_type == bf16:
            return True
    for vi in list(model.graph.input) + list(model.graph.output) + list(
        model.graph.value_info
    ):
        if vi.type.tensor_type.elem_type == bf16:
            return True
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == bf16:
                    return True
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR and attr.t.data_type == bf16:
                return True
            if attr.type == AttributeProto.TENSORS:
                if any(t.data_type == bf16 for t in attr.tensors):
                    return True
    return False


def _tensor_proto_bf16_to_fp32(tensor) -> bool:
    """In-place promote a TensorProto from BFLOAT16 to FLOAT. Returns True if changed."""
    from onnx import TensorProto, numpy_helper

    if tensor.data_type != TensorProto.BFLOAT16:
        return False
    arr = numpy_helper.to_array(tensor)
    name = tensor.name
    tensor.CopyFrom(
        numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)
    )
    return True


def _promote_bf16_onnx_to_fp32(model):
    """Rewrite BF16 weights/types to FP32 so ORT CPU can run the graph.

    BF16→FP32 is a lossless bit-width expand; runtime memory matches FP32.
    """
    from onnx import TensorProto, AttributeProto

    bf16 = TensorProto.BFLOAT16
    fp32 = TensorProto.FLOAT

    for init in model.graph.initializer:
        _tensor_proto_bf16_to_fp32(init)

    def _fix_type(type_proto) -> None:
        if type_proto.HasField("tensor_type") and type_proto.tensor_type.elem_type == bf16:
            type_proto.tensor_type.elem_type = fp32

    for vi in list(model.graph.input) + list(model.graph.output) + list(
        model.graph.value_info
    ):
        _fix_type(vi.type)

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == bf16:
                    attr.i = fp32
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR:
                _tensor_proto_bf16_to_fp32(attr.t)
            elif attr.type == AttributeProto.TENSORS:
                for t in attr.tensors:
                    _tensor_proto_bf16_to_fp32(t)

    # Drop stale dtype annotations so ORT re-infers from promoted tensors.
    del model.graph.value_info[:]
    return model


def _ort_model_source(model_path: str):
    """Return a path or in-memory proto for ORT.

    Disk may store BF16 LM / FineLocal weights; ORT CPU lacks many BF16 kernels,
    so promote those graphs to FP32 at load time.
    """
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
    return ort.InferenceSession(
        _ort_model_source(model_path), sess_options=so, providers=providers
    )


def build_speaker_prompt(text: str) -> str:
    placeholders = "".join(SPEAKER_PLACEHOLDER_TOKENS)
    return (
        f"{TARGET_TEXT_START_TOKEN}{text}{TARGET_TEXT_END_TOKEN}"
        f"{SPK_START_TOKEN}{placeholders}{SPK_END_TOKEN}"
        f"{TARGET_SPEECH_START_TOKEN}"
    )


class VoiceBank:
    """Voice bank + shared token embedding from Hojo-TTS-Light-40M-voice.npz.

    Required keys: ``voice_ids``, ``speaker_embeds``, ``speaker_vecs``,
    ``token_embedding`` (shared LM / FineLocal table, shape ``[V, H]``).
    """

    def __init__(self, voices_npz: str | Path) -> None:
        voices_npz = os.path.abspath(str(voices_npz))
        if not os.path.isfile(voices_npz):
            raise FileNotFoundError(f"Missing voices file: {voices_npz}")
        data = np.load(voices_npz, allow_pickle=True)
        if "voice_ids" not in data:
            raise ValueError(
                f"{voices_npz} must contain voice_ids; regenerate with export_speaker_onnx."
            )
        if "token_embedding" not in data:
            raise ValueError(
                f"{voices_npz} must contain token_embedding "
                f"(shared LM/FineLocal table); re-export or pack token_embedding.npy."
            )
        self.voice_ids = [str(v) for v in data["voice_ids"]]
        self.speaker_embeds = np.asarray(data["speaker_embeds"], dtype=np.float32)
        if "speaker_vecs" not in data:
            raise ValueError(
                f"{voices_npz} must contain speaker_vecs for FineLocal."
            )
        self.speaker_vecs = np.asarray(data["speaker_vecs"], dtype=np.float32)
        self.token_embedding = np.asarray(data["token_embedding"], dtype=np.float32)
        self._id_to_idx = {voice_id: idx for idx, voice_id in enumerate(self.voice_ids)}

    def list_voices(self) -> list[str]:
        return sorted(self.voice_ids)

    def get_speaker_embeds(self, voice: str) -> np.ndarray:
        voice_id = str(voice)
        if voice_id not in self._id_to_idx:
            raise KeyError(f"Unknown voice {voice_id!r}")
        idx = self._id_to_idx[voice_id]
        return self.speaker_embeds[idx : idx + 1].astype(np.float32, copy=False)

    def get_speaker_vec(self, voice: str) -> np.ndarray:
        voice_id = str(voice)
        if voice_id not in self._id_to_idx:
            raise KeyError(f"Unknown voice {voice_id!r}")
        idx = self._id_to_idx[voice_id]
        return self.speaker_vecs[idx : idx + 1].astype(np.float32, copy=False)


def _build_audio_token_id_to_code_table(tokenizer) -> np.ndarray:
    table = np.full((len(tokenizer),), -1, dtype=np.int64)
    for tid in range(len(tokenizer)):
        token_str = tokenizer.decode([tid], skip_special_tokens=False).strip()
        match = _AUDIO_TOKEN_REGEX.match(token_str)
        if match is not None:
            table[tid] = int(match.group(1))
    return table


def extract_audio_token_positions(
    generated_ids: np.ndarray,
    prompt_len: int,
    speech_end_id: int,
    id_to_code: np.ndarray,
) -> np.ndarray:
    ends = np.where(generated_ids == speech_end_id)[0]
    seq = generated_ids[: int(ends[0])] if ends.size else generated_ids
    valid = id_to_code[seq] >= 0
    rel = np.flatnonzero(valid)
    return (prompt_len + rel).astype(np.int64)


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


def empty_lm_past(
    num_layers: int, *, batch: int = 1, dtype=np.float32
) -> dict[str, np.ndarray]:
    past: dict[str, np.ndarray] = {}
    for layer in range(num_layers):
        past[f"past_key_values.{layer}.key"] = np.zeros((batch, 1, 0, 128), dtype=dtype)
        past[f"past_key_values.{layer}.value"] = np.zeros((batch, 1, 0, 128), dtype=dtype)
    return past


def inject_speaker_embeds(
    input_ids: np.ndarray,
    token_embeds: np.ndarray,
    speaker_embeds: np.ndarray,
    *,
    spk_start_id: int,
    num_slots: int = NUM_SPEAKER_INJECTION_SLOTS,
) -> np.ndarray:
    out = np.array(token_embeds, copy=True)
    batch, seq_len, _ = out.shape
    for b in range(batch):
        matches = np.where(input_ids[b] == spk_start_id)[0]
        if matches.size == 0:
            continue
        start = int(matches[0]) + 1
        for slot in range(num_slots):
            pos = start + slot
            if pos >= seq_len:
                break
            out[b, pos, :] = speaker_embeds[b, slot, :]
    return out


def _quantize_bits(binary_logits: np.ndarray) -> np.ndarray:
    """Match training hard quantization: logits > 0 → +1 else -1."""
    return np.where(binary_logits > 0.0, 1.0, -1.0).astype(np.float32)


class HojoTTSLightOnnx:
    """Low-level ONNX TTS runtime (unified LM + FineLocal + decoder + voices).

    Shared ``token_embedding`` inside ``Hojo-TTS-Light-40M-voice.npz`` is used for:
      - LM ``inputs_embeds`` (lookup + speaker inject in Python)
      - FineLocal ``coarse_embeddings``
    """

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
        lm_inputs = {inp.name for inp in self.lm.get_inputs()}
        if "inputs_embeds" not in lm_inputs:
            raise ValueError(f"{LM_ONNX_NAME} must expose inputs_embeds.")
        self._lm_output_names = [out.name for out in self.lm.get_outputs()]
        if "logits" not in self._lm_output_names:
            raise ValueError(f"{LM_ONNX_NAME} missing logits output.")
        if "last_hidden_state" not in self._lm_output_names:
            raise ValueError(f"{LM_ONNX_NAME} missing last_hidden_state output.")
        self._lm_logits_index = self._lm_output_names.index("logits")
        self._lm_hidden_index = self._lm_output_names.index("last_hidden_state")
        # Outputs are logits, last_hidden_state, then present.* KV tensors.
        self._lm_past_start = max(self._lm_logits_index, self._lm_hidden_index) + 1

        fine_local_path = os.path.join(self.models_dir, FINE_LOCAL_ONNX_NAME)
        if not os.path.isfile(fine_local_path):
            raise FileNotFoundError(f"Missing [{fine_local_path}].")
        self.fine_local = _build_ort_session(fine_local_path, provider, device_id, num_threads)
        fine_inputs = {inp.name for inp in self.fine_local.get_inputs()}
        if "coarse_embeddings" not in fine_inputs:
            raise ValueError(
                f"{FINE_LOCAL_ONNX_NAME} must expose coarse_embeddings "
                f"(shared token_embedding lookup from {VOICES_NPZ_NAME})."
            )

        self.voices = VoiceBank(voices_path)
        self.token_embedding = self.voices.token_embedding

        codec_path = os.path.join(self.models_dir, CODEC_ONNX_NAME)
        if not os.path.isfile(codec_path):
            raise FileNotFoundError(f"Missing [{codec_path}].")
        self.codec_decode = _build_ort_session(codec_path, provider, device_id, num_threads)

        self.tokenizer = _PromptTokenizer(self.models_dir)
        self.id_to_code = _build_audio_token_id_to_code_table(self.tokenizer)

        hop_length = 480
        codec_meta_path = os.path.join(self.models_dir, "codec_meta.json")
        if os.path.isfile(codec_meta_path):
            with open(codec_meta_path, encoding="utf-8") as f:
                codec_meta = json.load(f)
            hop_length = int(codec_meta.get("hop_length", hop_length))
            n_fft = int(codec_meta.get("n_fft", hop_length * 4))
        else:
            n_fft = hop_length * 4
        self.istft = ISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            padding="same",
        )

        with open(os.path.join(self.models_dir, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.num_layers = int(self.config["num_hidden_layers"])

        self.speech_end_id = self.tokenizer.convert_tokens_to_ids(TARGET_SPEECH_END_TOKEN)
        if self.speech_end_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer missing {TARGET_SPEECH_END_TOKEN!r}")
        self.spk_start_id = self.tokenizer.convert_tokens_to_ids(SPK_START_TOKEN)
        if self.spk_start_id == self.tokenizer.unk_token_id:
            raise ValueError(f"Tokenizer missing {SPK_START_TOKEN!r}")

    def _prepare_inputs_embeds(
        self,
        input_ids: np.ndarray,
        *,
        speaker_embeds: np.ndarray | None = None,
        inject_speaker: bool = False,
    ) -> np.ndarray:
        embeds = self.token_embedding[input_ids].astype(np.float32, copy=False)
        if inject_speaker:
            if speaker_embeds is None:
                raise ValueError("speaker_embeds required when inject_speaker=True")
            embeds = inject_speaker_embeds(
                input_ids,
                embeds,
                speaker_embeds.astype(np.float32, copy=False),
                spk_start_id=self.spk_start_id,
            )
        return embeds

    def _split_lm_outputs(
        self, outputs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Return (logits, last_hidden_state, past_kv_flat) from a unified LM run."""
        logits = outputs[self._lm_logits_index]
        hidden = outputs[self._lm_hidden_index]
        past = outputs[self._lm_past_start :]
        return logits, hidden, past

    @property
    def sample_rate(self) -> int:
        return OUTPUT_SAMPLE_RATE

    @property
    def available_voices(self) -> list[str]:
        return self.voices.list_voices()

    def _generate_coarse_tokens(
        self,
        input_ids: np.ndarray,
        speaker_embeds: np.ndarray,
        *,
        max_new_tokens: int,
        min_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        seq_len = int(input_ids.shape[1])
        position_ids = np.arange(seq_len, dtype=np.int64)[None, :]

        prefill_out = self.lm.run(
            None,
            {
                "inputs_embeds": self._prepare_inputs_embeds(
                    input_ids, speaker_embeds=speaker_embeds, inject_speaker=True
                ),
                "position_ids": position_ids,
                **empty_lm_past(self.num_layers),
            },
        )
        logits, hidden, past = self._split_lm_outputs(prefill_out)
        hidden_chunks: list[np.ndarray] = [hidden]

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

            step_ids = np.array([[next_token]], dtype=np.int64)
            feed = {
                "inputs_embeds": self._prepare_inputs_embeds(step_ids),
                "position_ids": np.array([[cur_len]], dtype=np.int64),
            }
            for layer in range(self.num_layers):
                feed[f"past_key_values.{layer}.key"] = past[layer * 2]
                feed[f"past_key_values.{layer}.value"] = past[layer * 2 + 1]

            decode_out = self.lm.run(None, feed)
            logits, hidden, past = self._split_lm_outputs(decode_out)
            hidden_chunks.append(hidden)
            next_token = sample_next_token(
                logits[0, -1],
                temperature=temperature,
                top_p=top_p,
                generated_ids=generated_ids,
                repetition_penalty=repetition_penalty,
            )
            generated_ids.append(next_token)
            cur_len += 1

        last_hidden = np.concatenate(hidden_chunks, axis=1)
        return np.array(generated_ids, dtype=np.int64), last_hidden

    def _bits_from_coarse(
        self,
        prompt_ids: np.ndarray,
        generated: np.ndarray,
        last_hidden: np.ndarray,
        speaker_vec: np.ndarray,
    ) -> np.ndarray:
        prompt_len = int(prompt_ids.shape[1])
        full_ids = np.concatenate([prompt_ids[0], generated], axis=0)[None, :].astype(
            np.int64
        )
        audio_positions = extract_audio_token_positions(
            generated, prompt_len, self.speech_end_id, self.id_to_code
        )
        if audio_positions.size == 0:
            raise RuntimeError("LM did not generate valid audio tokens like [123].")
        if np.any(audio_positions < 1):
            raise RuntimeError("Audio token has no previous hidden state.")
        needed = int(audio_positions.max()) - 1
        if needed >= last_hidden.shape[1]:
            raise RuntimeError(
                f"Cached LM hidden length {last_hidden.shape[1]} is shorter than "
                f"required index {needed}."
            )

        hidden_states = last_hidden[:, audio_positions - 1, :].astype(np.float32)
        coarse_token_ids = full_ids[:, audio_positions].astype(np.int64)
        valid_mask = np.ones((1, int(audio_positions.size)), dtype=bool)

        logits = self.fine_local.run(
            None,
            {
                "hidden_states": hidden_states,
                "coarse_embeddings": self.token_embedding[coarse_token_ids],
                "speaker_embedding": speaker_vec.astype(np.float32),
                "valid_mask": valid_mask,
            },
        )[0]
        return _quantize_bits(logits[0])

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

        speaker_embeds = self.voices.get_speaker_embeds(voice)
        speaker_vec = self.voices.get_speaker_vec(voice)

        generated, last_hidden = self._generate_coarse_tokens(
            input_ids,
            speaker_embeds,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        bits = self._bits_from_coarse(input_ids, generated, last_hidden, speaker_vec)
        mag, phase = self.codec_decode.run(None, {"bits": bits})
        return wav_from_mag_phase(mag, phase, self.istft)
