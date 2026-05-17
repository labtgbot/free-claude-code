"""Voice note transcription for messaging platforms.

Supports:
- Local Whisper (cpu/cuda): Hugging Face transformers pipeline
- NVIDIA NIM: NVIDIA NIM Whisper/Parakeet
"""

from pathlib import Path
from typing import Any

from loguru import logger

from providers.nvidia_nim.voice import (
    transcribe_audio_file as transcribe_nvidia_nim_audio,
)

# Max file size in bytes (25 MB)
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024

# Short model names -> full Hugging Face model IDs (for local Whisper)
_MODEL_MAP: dict[str, str] = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}

_MODEL_REVISIONS: dict[str, str] = {
    "openai/whisper-tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "openai/whisper-base": "e37978b90ca9030d5170a5c07aadb050351a65bb",
    "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
    "openai/whisper-medium": "abdf7c39ab9d0397620ccaea8974cc764cd0953e",
    "openai/whisper-large-v2": "ae4642769ce2ad8fc292556ccea8e901f1530655",
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "openai/whisper-large-v3-turbo": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
}

# Lazy-loaded pipelines: (model_id, device, hf_token_fingerprint, revision) -> pipeline
_pipeline_cache: dict[tuple[str, str, str, str], Any] = {}


def _resolve_model_id(whisper_model: str) -> str:
    """Resolve short name to full Hugging Face model ID."""
    return _MODEL_MAP.get(whisper_model, whisper_model)


def _resolve_model_revision(model_id: str, hf_model_revision: str = "") -> str:
    """Resolve the immutable Hugging Face revision for a local Whisper model."""
    if hf_model_revision:
        return hf_model_revision
    if revision := _MODEL_REVISIONS.get(model_id):
        return revision
    raise ValueError(
        "HF_MODEL_REVISION is required when WHISPER_MODEL uses a custom "
        "Hugging Face model ID for local transcription."
    )


def _get_pipeline(
    model_id: str,
    device: str,
    hf_token: str = "",
    hf_model_revision: str = "",
) -> Any:
    """Lazy-load transformers Whisper pipeline. Raises ImportError if not installed."""
    global _pipeline_cache
    if device not in ("cpu", "cuda"):
        raise ValueError(f"whisper_device must be 'cpu' or 'cuda', got {device!r}")
    resolved_token = hf_token or ""
    model_revision = _resolve_model_revision(model_id, hf_model_revision)
    cache_key = (model_id, device, resolved_token, model_revision)
    if cache_key not in _pipeline_cache:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

            hf_auth_token = resolved_token or None

            use_cuda = device == "cuda" and torch.cuda.is_available()
            pipe_device = "cuda:0" if use_cuda else "cpu"
            model_dtype = torch.float16 if use_cuda else torch.float32

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                dtype=model_dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                token=hf_auth_token,
                revision=model_revision,
            )
            model = model.to(pipe_device)
            processor = AutoProcessor.from_pretrained(
                model_id,
                token=hf_auth_token,
                revision=model_revision,
            )

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                device=pipe_device,
            )
            _pipeline_cache[cache_key] = pipe
            logger.debug(
                f"Loaded Whisper pipeline: model={model_id} device={pipe_device}"
            )
        except ImportError as e:
            raise ImportError(
                "Local Whisper requires the voice_local extra. Install with: uv sync --extra voice_local"
            ) from e
    return _pipeline_cache[cache_key]


def transcribe_audio(
    file_path: Path,
    mime_type: str,
    *,
    whisper_model: str = "base",
    whisper_device: str = "cpu",
    hf_token: str = "",
    hf_model_revision: str = "",
    nvidia_nim_api_key: str = "",
) -> str:
    """
    Transcribe audio file to text.

    Supports:
    - whisper_device="cpu"/"cuda": local Whisper (requires voice_local extra)
    - whisper_device="nvidia_nim": NVIDIA NIM Whisper API (requires voice extra)

    Args:
        file_path: Path to audio file (OGG, MP3, MP4, WAV, M4A supported)
        mime_type: MIME type of the audio (e.g. "audio/ogg")
        whisper_model: Model ID or short name (local) or NVIDIA NIM model
        whisper_device: "cpu" | "cuda" | "nvidia_nim"

    Returns:
        Transcribed text

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file too large
        ImportError: If voice_local extra not installed (for local Whisper)
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    size = file_path.stat().st_size
    if size > MAX_AUDIO_SIZE_BYTES:
        raise ValueError(
            f"Audio file too large ({size} bytes). Max {MAX_AUDIO_SIZE_BYTES} bytes."
        )

    if whisper_device == "nvidia_nim":
        return transcribe_nvidia_nim_audio(
            file_path, whisper_model, api_key=nvidia_nim_api_key
        )
    return _transcribe_local(
        file_path,
        whisper_model,
        whisper_device,
        hf_token=hf_token,
        hf_model_revision=hf_model_revision,
    )


# Whisper expects 16 kHz sample rate
_WHISPER_SAMPLE_RATE = 16000


def _load_audio(file_path: Path) -> dict[str, Any]:
    """Load audio file to waveform dict. No ffmpeg required."""
    import librosa

    waveform, sr = librosa.load(str(file_path), sr=_WHISPER_SAMPLE_RATE, mono=True)
    return {"array": waveform, "sampling_rate": sr}


def _transcribe_local(
    file_path: Path,
    whisper_model: str,
    whisper_device: str,
    *,
    hf_token: str = "",
    hf_model_revision: str = "",
) -> str:
    """Transcribe using transformers Whisper pipeline."""
    model_id = _resolve_model_id(whisper_model)
    pipe = _get_pipeline(
        model_id,
        whisper_device,
        hf_token=hf_token,
        hf_model_revision=hf_model_revision,
    )
    audio = _load_audio(file_path)
    result = pipe(audio, generate_kwargs={"language": "en", "task": "transcribe"})
    text = result.get("text", "") or ""
    if isinstance(text, list):
        text = " ".join(text) if text else ""
    result_text = text.strip()
    logger.debug(f"Local transcription: {len(result_text)} chars")
    return result_text or "(no speech detected)"
