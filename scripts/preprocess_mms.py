"""
Preprocessing for MMS-VITS Turkish fine-tuning (v2 — saves raw waveforms).

Loads afkfatih/turkish-tts-combined-raw (81K samples, 7 Turkish speakers),
resamples audio to 16000 Hz (MMS-TTS native rate), saves raw waveforms + tokenizes text, and expands
the English MMS tokenizer vocabulary with Turkish-specific characters.

Raw waveforms are required by the full VITS GAN trainer (posterior encoder +
discriminator both operate on raw audio).

Output layout:
    /data/tts/mms-dataset/
    ├── train/          HF Dataset (Arrow) — input_ids, waveform, text
    ├── validation/     HF Dataset (Arrow)
    └── tokenizer/      Updated VitsTokenizer with Turkish vocab

Usage:
    python preprocess_mms.py --out_dir /data/tts/mms-dataset
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import librosa
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# MMS-VITS model constants — must match facebook/mms-tts-eng config
TARGET_SR = 16_000  # MMS-TTS native sample rate
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0
FMAX = None

MAX_DURATION_S = 10.0   # cap at 10 s to bound waveform storage
MIN_DURATION_S = 0.5

# Characters present in Turkish but missing from the English ASCII tokenizer
TURKISH_EXTRA_CHARS = list("çğışöüÇĞİŞÖÜ")

SENTINEL_VERSION = "v3-16khz"  # bumped: TARGET_SR changed to 16000


def resample_waveform(waveform: np.ndarray, sr: int) -> np.ndarray:
    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
    return waveform.astype(np.float32)


def process_sample(batch, tokenizer):
    audio = batch["audio"]
    # datasets v4+ with Audio(decode=False) returns raw bytes — decode manually
    if isinstance(audio, dict) and "bytes" in audio and audio["bytes"]:
        import io
        import soundfile as sf
        waveform, sr = sf.read(io.BytesIO(audio["bytes"]))
        waveform = waveform.astype(np.float32)
    else:
        waveform = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)

    duration = len(waveform) / sr
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        batch["valid"]     = False
        batch["input_ids"] = []
        batch["waveform"]  = []
        batch["duration"]  = duration
        return batch

    waveform = resample_waveform(waveform, sr)
    # Truncate to MAX_DURATION_S after resampling
    max_samples = int(TARGET_SR * MAX_DURATION_S)
    waveform = waveform[:max_samples]

    text = batch["text"].strip().lower()
    encoded = tokenizer(text, return_tensors=None)

    batch["input_ids"] = encoded["input_ids"]
    batch["waveform"]  = waveform.tolist()   # list[float] — stored as Arrow array
    batch["duration"]  = float(len(waveform) / TARGET_SR)
    batch["valid"]     = True
    return batch


def expand_tokenizer_for_turkish(tokenizer):
    """Add Turkish-specific characters missing from the English vocab."""
    vocab = tokenizer.get_vocab()
    missing = [c for c in TURKISH_EXTRA_CHARS if c not in vocab]
    if missing:
        tokenizer.add_tokens(missing)
        log.info(f"Added {len(missing)} Turkish characters to tokenizer: {missing}")
    else:
        log.info("All Turkish characters already in tokenizer vocab.")
    return tokenizer, missing


SENTINEL = ".preprocessed"


def main():
    p = argparse.ArgumentParser(description="Preprocess Turkish TTS dataset for MMS-VITS")
    p.add_argument("--out_dir", type=Path, default=Path("/data/tts/mms-dataset"))
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size for smoke tests (e.g. 500)")
    p.add_argument("--val_split", type=float, default=0.05)
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--hf_cache", type=str, default="/data/tts/hf-cache")
    p.add_argument("--base_model", type=str, default="facebook/mms-tts-eng")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if preprocessed data already exists")
    args = p.parse_args()

    os.environ["HF_HOME"] = args.hf_cache
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sentinel = args.out_dir / SENTINEL
    if not args.force and sentinel.exists() and (args.out_dir / "train").exists():
        # Re-run if older sentinel version (missing waveforms)
        sentinel_txt = sentinel.read_text()
        if SENTINEL_VERSION not in sentinel_txt:
            log.info("Sentinel is v1 (no waveforms) — re-running preprocessing for v2.")
        else:
            log.info(
                f"Preprocessed dataset found at {args.out_dir} (sentinel: {sentinel}). "
                "Skipping — pass --force to re-run."
            )
            return

    from datasets import load_dataset
    from transformers import VitsTokenizer

    log.info(f"Loading tokenizer: {args.base_model}")
    tokenizer = VitsTokenizer.from_pretrained(args.base_model)
    tokenizer, added_tokens = expand_tokenizer_for_turkish(tokenizer)

    tokenizer_path = args.out_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_path))
    log.info(f"Tokenizer saved → {tokenizer_path}  (vocab size: {len(tokenizer)})")

    log.info("Loading afkfatih/turkish-tts-combined-raw...")
    from datasets import Audio
    ds = load_dataset("afkfatih/turkish-tts-combined-raw", split="train",
                      cache_dir=args.hf_cache)
    # datasets v4+ requires torchcodec for auto-decode — bypass with raw bytes
    ds = ds.cast_column("audio", Audio(decode=False))
    log.info(f"Total samples: {len(ds)}")

    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        log.info(f"Capped to {len(ds)} samples")

    # Save tokenizer path so worker processes can reload it independently
    _tmp_tok_path = str(args.out_dir / "tokenizer")

    def process_with_reloaded_tokenizer(batch):
        from transformers import VitsTokenizer as _VTok
        tok = _VTok.from_pretrained(_tmp_tok_path)
        return process_sample(batch, tok)

    log.info("Processing samples (tokenize + mel spectrogram)...")
    ds = ds.map(
        process_with_reloaded_tokenizer,
        remove_columns=["audio"],
        num_proc=args.num_proc,
        desc="tokenize+mel",
    )

    before = len(ds)
    ds = ds.filter(lambda x: x["valid"])
    log.info(f"Valid samples: {len(ds)}/{before}")

    keep = {"input_ids", "waveform", "text"}
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

    log.info("Splitting train/validation...")
    splits = ds.train_test_split(test_size=args.val_split, seed=42)
    train_ds = splits["train"]
    val_ds = splits["test"]
    log.info(f"Train: {len(train_ds)}  |  Validation: {len(val_ds)}")

    train_ds.save_to_disk(str(args.out_dir / "train"))
    val_ds.save_to_disk(str(args.out_dir / "validation"))

    import datetime
    stats = (
        f"sentinel_version: {SENTINEL_VERSION}\n"
        f"dataset: afkfatih/turkish-tts-combined-raw\n"
        f"base_model: {args.base_model}\n"
        f"train_samples: {len(train_ds)}\n"
        f"val_samples: {len(val_ds)}\n"
        f"sample_rate: {TARGET_SR}\n"
        f"max_duration_s: {MAX_DURATION_S}\n"
        f"columns: input_ids, waveform, text\n"
        f"added_tokens: {added_tokens}\n"
        f"tokenizer_vocab_size: {len(tokenizer)}\n"
        f"max_samples_cap: {args.max_samples}\n"
        f"completed_at: {datetime.datetime.utcnow().isoformat()}Z\n"
    )
    (args.out_dir / "stats.txt").write_text(stats)
    sentinel.write_text(stats)
    log.info(f"Preprocessing complete. Sentinel written → {sentinel}")


if __name__ == "__main__":
    main()
