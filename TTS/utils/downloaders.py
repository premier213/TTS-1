import os
import csv
from typing import Dict, Any, List, Tuple

import numpy as np
from datasets import load_dataset
import soundfile as sf


TEXT_CANDIDATE_KEYS: List[str] = [
    "text",
    "sentence",
    "transcript",
    "normalized_text",
    "prompt",
    "label",
]


def _normalize_keys(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict with stripped keys (HF datasets sometimes contain whitespace)."""
    return {str(k).strip(): v for k, v in sample.items()}


def _pick_text_key(
    sample: Dict[str, Any], candidates: List[str] = TEXT_CANDIDATE_KEYS
) -> str:
    """Choose the first candidate key present in sample that contains a string."""
    sample = _normalize_keys(sample)
    for key in candidates:
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            return key
    raise KeyError(
        f"No usable text field found. Available keys: {list(sample.keys())}. "
        f"Tried: {candidates}"
    )


def _to_int16_audio(audio_array: np.ndarray) -> np.ndarray:
    """
    Convert audio to int16 PCM, clipping floats to [-1, 1] first.
    Returns shape (frames, channels) for soundfile compatibility.
    """
    x = np.asarray(audio_array)

    # Guard against empty/bad audio
    if x.ndim == 0 or x.size == 0:
        raise ValueError("Empty audio array")

    # Ensure shape (frames, channels)
    # HF audio commonly returns mono as (frames,), stereo as (frames, channels)
    if x.ndim == 1:
        x = x[:, None]
    elif x.ndim == 2:
        pass
    else:
        # Sometimes weird shapes show up; try flattening to mono safely
        x = x.reshape(-1, 1)

    # Convert dtype
    if np.issubdtype(x.dtype, np.floating):
        x = np.clip(x, -1.0, 1.0)
        x = (x * 32767.0).round().astype(np.int16)
    elif x.dtype != np.int16:
        # If it's int32/int64/etc, clip to int16 range
        x = np.clip(x, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)

    return x


def download_HAL_9000_Speech(dataset_root: str) -> None:
    """
    Download 'campwill/HAL-9000-Speech' from Hugging Face and write it in a TTS-friendly layout:

      dataset_root/
        metadata.csv          (wavs/xxxxx.wav|text|speaker)
        wavs/
          00000.wav
          00001.wav
          ...

    Supports HF Datasets where `audio` is a torchcodec AudioDecoder.
    """
    import os
    import csv
    import numpy as np
    import soundfile as sf
    from datasets import load_dataset

    TEXT_CANDIDATES = [
        "text",
        "sentence",
        "transcript",
        "normalized_text",
        "prompt",
        "label",
    ]

    def normalize_keys(sample: dict) -> dict:
        return {str(k).strip(): v for k, v in sample.items()}

    def find_text_key(sample: dict) -> str:
        sample = normalize_keys(sample)
        for k in TEXT_CANDIDATES:
            v = sample.get(k)
            if isinstance(v, str) and v.strip():
                return k
        raise KeyError(
            f"No usable text field found. Keys={list(sample.keys())}, tried={TEXT_CANDIDATES}"
        )

    ds = load_dataset("campwill/HAL-9000-Speech")

    os.makedirs(dataset_root, exist_ok=True)
    wav_dir = os.path.join(dataset_root, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    meta_path = os.path.join(dataset_root, "metadata.csv")

    first = normalize_keys(ds["train"][0])
    text_key = find_text_key(first)
    print("Using text key:", text_key)
    print("Train columns:", ds["train"].column_names)

    written = 0
    skipped = 0
    reasons = {}

    def _skip(reason: str):
        nonlocal skipped
        skipped += 1
        reasons[reason] = reasons.get(reason, 0) + 1

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|")

        for i, sample in enumerate(ds["train"]):
            sample = normalize_keys(sample)

            text_val = sample.get(text_key, "")
            audio_obj = sample.get("audio")

            if not isinstance(text_val, str) or not text_val.strip():
                _skip("missing_text")
                continue
            if audio_obj is None:
                _skip("missing_audio")
                continue

            text = text_val.strip().replace("\n", " ")

            # --- Decode audio ---
            try:
                # HF Datasets v4 torchcodec AudioDecoder
                if hasattr(audio_obj, "get_all_samples"):
                    decoded = audio_obj.get_all_samples()
                    sr = int(decoded.sample_rate)
                    x = decoded.data  # often torch.Tensor (channels, time)
                    if hasattr(x, "cpu"):
                        x = x.cpu().numpy()
                    else:
                        x = np.asarray(x)

                # Older HF format: dict with array + sampling_rate
                elif (
                    isinstance(audio_obj, dict)
                    and "array" in audio_obj
                    and "sampling_rate" in audio_obj
                ):
                    sr = int(audio_obj["sampling_rate"])
                    x = np.asarray(audio_obj["array"])

                else:
                    _skip("unknown_audio_type")
                    continue
            except Exception:
                _skip("decode_failed")
                continue

            if x.ndim == 0 or x.size == 0:
                _skip("empty_audio")
                continue

            # Shape for soundfile: (time, channels)
            # torchcodec often gives (channels, time)
            if x.ndim == 2:
                if x.shape[0] in (1, 2) and x.shape[1] > x.shape[0]:
                    x = x.T
            elif x.ndim == 1:
                x = x[:, None]
            else:
                x = x.reshape(-1, 1)

            # Convert to int16 PCM
            if np.issubdtype(x.dtype, np.floating):
                x = np.clip(x, -1.0, 1.0)
                x = (x * 32767.0).round().astype(np.int16)
            elif x.dtype != np.int16:
                x = np.clip(x, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(
                    np.int16
                )

            filename = f"{i:05d}.wav"
            wav_path = os.path.join(wav_dir, filename)

            try:
                sf.write(wav_path, x, sr, subtype="PCM_16")
            except Exception:
                _skip("write_failed")
                continue

            writer.writerow([f"wavs/{filename}", text, "none"])
            written += 1

    print(f"HAL-9000-Speech: wrote {written} samples, skipped {skipped}")
    if skipped:
        print("Skip reasons:", reasons)
    print(f"Dataset root: {dataset_root}")
    print(f"Metadata: {meta_path}")
