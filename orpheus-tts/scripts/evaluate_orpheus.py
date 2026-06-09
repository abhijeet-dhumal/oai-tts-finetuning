"""
Post-training evaluation for Orpheus-3B Turkish TTS.

Computes and logs to MLflow:
  - WER  / CER      via OpenAI Whisper ASR (intelligibility)
  - UTMOS            via UTMOS-strong neural MOS predictor (naturalness, 1–5)
  - RTF              real-time factor (inference speed)
  - Audio WAVs + spectrograms for each test sentence
  - Summary comparison table (baseline pretrained vs fine-tuned)

Usage
-----
  python evaluate_orpheus.py                    # uses env vars
  python evaluate_orpheus.py --help

Runs attach to the existing MLflow experiment so metrics appear alongside
the training run for direct comparison in the MLflow UI.
"""

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# ── CLI / env config ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--finetuned_model",  default=os.environ.get("FINETUNED_MODEL",  "/data/orpheus/checkpoints/turkish/final"))
    p.add_argument("--base_model",       default=os.environ.get("BASE_MODEL",       "canopylabs/orpheus-tts-0.1-pretrained"))
    p.add_argument("--hf_cache",         default=os.environ.get("HF_HOME",          "/data/orpheus/hf-cache"))
    p.add_argument("--whisper_model",    default=os.environ.get("WHISPER_MODEL",    "large-v3"))
    p.add_argument("--mlflow_run_id",    default=os.environ.get("MLFLOW_RUN_ID",    ""),     help="Attach to existing training run")
    p.add_argument("--mlflow_experiment",default=os.environ.get("MLFLOW_EXPERIMENT","orpheus-turkish-tts"))
    p.add_argument("--output_dir",       default=os.environ.get("EVAL_OUTPUT_DIR",  "/data/orpheus/eval"))
    return p.parse_args()


# ── Orpheus SNAC token constants ──────────────────────────────────────────────

LLAMA_VOCAB  = 128_256
CODE_OFFSET  = LLAMA_VOCAB + 10
N_CODEBOOK   = 4_096
N_PER_FRAME  = 7
SNAC_SR      = 24_000

TOK_SOH = LLAMA_VOCAB + 3
TOK_EOH = LLAMA_VOCAB + 4
TOK_SOA = LLAMA_VOCAB + 5
TOK_EOA = LLAMA_VOCAB + 6
TOK_SOS = LLAMA_VOCAB + 1
TOK_EOT = LLAMA_VOCAB + 9

# ── Evaluation sentences ──────────────────────────────────────────────────────
# 10 diverse sentences to give statistically meaningful WER/UTMOS averages

EVAL_SET = [
    ("flight_announce",  "sayın yolcularımız, uçuşumuz yaklaşık iki saat sürecektir."),
    ("welcome",          "istanbul'a hoş geldiniz."),
    ("safety",           "güvenlik nedeniyle elektronik cihazlarınızı kapalı tutunuz."),
    ("farewell",         "teşekkür ederiz, iyi yolculuklar dileriz."),
    ("weather",          "bugün istanbul'da hava bulutlu ve serin olacak."),
    ("news_intro",       "ana haberlere geçmeden önce önemli bir duyurumuz var."),
    ("directions",       "düz gidin, sonra sağa dönün ve köprüyü geçin."),
    ("question",         "yarın toplantıya katılabilir misiniz?"),
    ("tech",             "yapay zeka teknolojisi her geçen gün gelişmeye devam ediyor."),
    ("emergency",        "acil durum! lütfen binayı derhal tahliye edin."),
]


# ── Audio helpers ─────────────────────────────────────────────────────────────

def build_prompt(tokenizer, text: str) -> list:
    ids = tokenizer.encode(text, add_special_tokens=True) + [TOK_EOT]
    return [TOK_SOH] + ids + [TOK_EOH, TOK_SOA, TOK_SOS]


def snac_decode(snac_model, token_ids: list, device) -> np.ndarray | None:
    audio_ids = [t for t in token_ids if t >= CODE_OFFSET]
    n = len(audio_ids) // N_PER_FRAME
    if n == 0:
        return None
    audio_ids = audio_ids[:n * N_PER_FRAME]

    l0, l1, l2 = [], [], []
    for f in range(n):
        g = audio_ids[N_PER_FRAME * f : N_PER_FRAME * (f + 1)]
        l0.append((g[0] - CODE_OFFSET) % N_CODEBOOK)
        l1.append((g[1] - CODE_OFFSET) % N_CODEBOOK)
        l2.append((g[2] - CODE_OFFSET) % N_CODEBOOK)
        l2.append((g[3] - CODE_OFFSET) % N_CODEBOOK)
        l1.append((g[4] - CODE_OFFSET) % N_CODEBOOK)
        l2.append((g[5] - CODE_OFFSET) % N_CODEBOOK)
        l2.append((g[6] - CODE_OFFSET) % N_CODEBOOK)

    def _t(x): return torch.tensor(x, dtype=torch.long).unsqueeze(0).to(device)
    wav = snac_model.decode([_t(l0), _t(l1), _t(l2)])
    return wav.squeeze().cpu().float().numpy()


def generate_audio(model, tokenizer, snac_model, text: str, device) -> tuple:
    t0 = time.perf_counter()
    prompt = build_prompt(tokenizer, text)
    inp = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.generate(
            inp, max_new_tokens=1200, do_sample=True,
            temperature=0.7, repetition_penalty=1.1, eos_token_id=TOK_EOA,
        )
    elapsed = time.perf_counter() - t0
    wav = snac_decode(snac_model, out[0][len(prompt):].cpu().tolist(), device)
    return wav, elapsed


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_wer_cer(whisper_model, wav: np.ndarray, reference: str) -> tuple[float, float]:
    """Run Whisper ASR on wav; compute WER and CER against reference text."""
    import jiwer

    result = whisper_model.transcribe(
        wav.astype(np.float32),
        language="tr",
        task="transcribe",
    )
    hypothesis = result["text"].strip().lower()
    ref = reference.strip().lower()

    wer = jiwer.wer(ref, hypothesis)
    cer = jiwer.cer(ref, hypothesis)
    return wer, cer, hypothesis


def compute_utmos(utmos_predictor, wav: np.ndarray) -> float:
    """
    Predict MOS score (1–5) using UTMOS-strong.
    Higher = more natural sounding.
    """
    import torchaudio
    wav_tensor = torch.tensor(wav).unsqueeze(0)
    # UTMOS expects 16kHz; resample from 24kHz
    wav_16k = torchaudio.functional.resample(wav_tensor, SNAC_SR, 16_000)
    score = utmos_predictor.predict_from_wavs([wav_16k])
    return float(score[0])


def log_spectrogram(wav: np.ndarray, title: str, path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.specgram(wav, Fs=SNAC_SR, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(str(path), dpi=100)
    plt.close(fig)


def log_results_table(rows: list[dict], path: Path):
    """Write a markdown + PNG comparison table to disk."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_markdown(str(path.with_suffix(".md")), index=False, floatfmt=".3f")

    # Render as a PNG table for direct embedding in the blog post
    fig, ax = plt.subplots(figsize=(14, 0.5 + 0.4 * len(df)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(str(path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Model loader helper ───────────────────────────────────────────────────────

def load_orpheus(model_path: str, hf_cache: str, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=hf_cache)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=hf_cache,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval().to(device)
    return model, tokenizer


# ── Core evaluation loop ──────────────────────────────────────────────────────

def evaluate_model(
    model, tokenizer, snac_model, whisper_model, utmos_predictor,
    label: str, device, out_dir: Path,
) -> list[dict]:
    """
    Run full evaluation on EVAL_SET.
    Returns list of per-sentence result dicts.
    """
    import soundfile as sf

    results = []
    for slug, text in EVAL_SET:
        print(f"  [{label}] {slug} …", flush=True)
        wav, elapsed = generate_audio(model, tokenizer, snac_model, text, device)

        row = {"model": label, "sentence": slug, "text": text}

        if wav is None:
            row.update({"wer": 1.0, "cer": 1.0, "utmos": 1.0, "rtf": 0.0, "transcript": ""})
            results.append(row)
            continue

        duration = len(wav) / SNAC_SR
        row["rtf"] = elapsed / max(duration, 1e-6)

        # WER / CER
        wer, cer, transcript = compute_wer_cer(whisper_model, wav, text)
        row["wer"]        = wer
        row["cer"]        = cer
        row["transcript"] = transcript

        # UTMOS MOS
        if utmos_predictor is not None:
            try:
                row["utmos"] = compute_utmos(utmos_predictor, wav)
            except Exception as e:
                print(f"    UTMOS failed: {e}", flush=True)
                row["utmos"] = float("nan")
        else:
            row["utmos"] = float("nan")

        # Save WAV + spectrogram
        wav_path = out_dir / f"{label}_{slug}.wav"
        sf.write(str(wav_path), wav, samplerate=SNAC_SR)
        spec_path = out_dir / f"{label}_{slug}_spec.png"
        log_spectrogram(wav, f"{label} | {slug}: {text[:50]}", spec_path)

        results.append(row)

    return results


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_eval_to_mlflow(mlflow, results: list[dict], out_dir: Path, label: str):
    """Log per-sentence metrics, aggregate metrics, and artifact files."""
    import pandas as pd

    df = pd.DataFrame(results)
    model_df = df[df["model"] == label]

    # Aggregate metrics
    mlflow.log_metrics({
        f"eval/{label}/wer_mean":   model_df["wer"].mean(),
        f"eval/{label}/cer_mean":   model_df["cer"].mean(),
        f"eval/{label}/utmos_mean": model_df["utmos"].mean(),
        f"eval/{label}/rtf_mean":   model_df["rtf"].mean(),
    })

    # Per-sentence metrics
    for _, row in model_df.iterrows():
        slug = row["sentence"]
        mlflow.log_metrics({
            f"eval/{label}/{slug}/wer":   row["wer"],
            f"eval/{label}/{slug}/cer":   row["cer"],
            f"eval/{label}/{slug}/utmos": row["utmos"],
        })

    # Log WAVs and spectrograms
    for f in out_dir.glob(f"{label}_*.wav"):
        mlflow.log_artifact(str(f), artifact_path=f"eval/{label}/audio")
    for f in out_dir.glob(f"{label}_*_spec.png"):
        mlflow.log_artifact(str(f), artifact_path=f"eval/{label}/spectrograms")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import mlflow
    import soundfile as sf

    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}", flush=True)
    print(f"Fine-tuned model: {args.finetuned_model}", flush=True)
    print(f"Base model (baseline): {args.base_model}", flush=True)

    # ── Load SNAC codec ───────────────────────────────────────────────────────
    print("Loading SNAC-24kHz …", flush=True)
    from snac import SNAC
    snac_model = SNAC.from_pretrained(
        "hubertsiuzdak/snac_24khz", cache_dir=args.hf_cache
    ).eval().to(device)

    # ── Load Whisper (ASR for WER/CER) ────────────────────────────────────────
    print(f"Loading Whisper {args.whisper_model} …", flush=True)
    import whisper
    whisper_model = whisper.load_model(args.whisper_model, device=str(device))

    # ── Load UTMOS (neural MOS predictor) ─────────────────────────────────────
    utmos_predictor = None
    try:
        print("Loading UTMOS-strong …", flush=True)
        import utmos
        utmos_predictor = utmos.Score(device=str(device))
        print("UTMOS loaded.", flush=True)
    except Exception as e:
        print(f"UTMOS unavailable ({e}) — skipping MOS scores.", flush=True)

    # ── Load both models ──────────────────────────────────────────────────────
    print("Loading fine-tuned model …", flush=True)
    ft_model, ft_tokenizer = load_orpheus(args.finetuned_model, args.hf_cache, device)

    print("Loading baseline (pretrained) model …", flush=True)
    base_model, base_tokenizer = load_orpheus(args.base_model, args.hf_cache, device)

    # ── MLflow run ────────────────────────────────────────────────────────────
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment

    run_ctx = (
        mlflow.start_run(run_id=args.mlflow_run_id)
        if args.mlflow_run_id
        else mlflow.start_run(run_name="eval-orpheus-tr", nested=False)
    )

    with run_ctx as run:
        print(f"MLflow run: {run.info.run_id}", flush=True)

        # ── Evaluate both models ──────────────────────────────────────────────
        print("\n=== Evaluating BASELINE (pretrained) ===", flush=True)
        base_results = evaluate_model(
            base_model, base_tokenizer, snac_model,
            whisper_model, utmos_predictor,
            label="baseline", device=device, out_dir=out_dir,
        )
        log_eval_to_mlflow(mlflow, base_results, out_dir, "baseline")

        print("\n=== Evaluating FINE-TUNED ===", flush=True)
        ft_results = evaluate_model(
            ft_model, ft_tokenizer, snac_model,
            whisper_model, utmos_predictor,
            label="finetuned", device=device, out_dir=out_dir,
        )
        log_eval_to_mlflow(mlflow, ft_results, out_dir, "finetuned")

        # ── Summary comparison table ──────────────────────────────────────────
        import pandas as pd
        all_results = base_results + ft_results
        df = pd.DataFrame(all_results)

        summary = (
            df.groupby("model")[["wer", "cer", "utmos", "rtf"]]
            .mean()
            .reset_index()
            .rename(columns={
                "model": "Model",
                "wer":   "WER ↓",
                "cer":   "CER ↓",
                "utmos": "UTMOS ↑",
                "rtf":   "RTF ↓",
            })
        )

        # Delta row: improvement from baseline → fine-tuned
        b = summary[summary["Model"] == "baseline"].iloc[0]
        f = summary[summary["Model"] == "finetuned"].iloc[0]
        delta_row = pd.DataFrame([{
            "Model":    "Δ (improvement)",
            "WER ↓":    round(b["WER ↓"] - f["WER ↓"], 3),
            "CER ↓":    round(b["CER ↓"] - f["CER ↓"], 3),
            "UTMOS ↑":  round(f["UTMOS ↑"] - b["UTMOS ↑"], 3),
            "RTF ↓":    round(b["RTF ↓"] - f["RTF ↓"], 3),
        }])
        summary = pd.concat([summary, delta_row], ignore_index=True)

        print("\n=== Evaluation Summary ===", flush=True)
        print(summary.to_string(index=False), flush=True)

        table_path = out_dir / "eval_summary"
        log_results_table(summary.to_dict("records"), table_path)
        mlflow.log_artifact(str(table_path.with_suffix(".md")),   artifact_path="eval")
        mlflow.log_artifact(str(table_path.with_suffix(".png")),  artifact_path="eval")

        # Log aggregate comparison metrics for dashboard charts
        mlflow.log_metrics({
            "eval/wer_improvement":   b["WER ↓"]  - f["WER ↓"],
            "eval/cer_improvement":   b["CER ↓"]  - f["CER ↓"],
            "eval/utmos_improvement": f["UTMOS ↑"] - b["UTMOS ↑"],
        })

        print(f"\nEvaluation complete. Results at: {out_dir}", flush=True)
        print(f"MLflow run: {run.info.run_id}", flush=True)


if __name__ == "__main__":
    main()
