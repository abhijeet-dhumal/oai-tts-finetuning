"""
3-way Turkish TTS evaluation with full MLflow artifact tracking.

Compares:
  baseline  — facebook/mms-tts-eng  (original tokenizer, Turkish chars mangled)
  finetuned — our trained checkpoint (full VITS GAN, Stage 2)
  reference — facebook/mms-tts-tur  (gold standard)

MLflow artifacts logged per run:
  audio/       — all WAV samples (every sentence)
  spectrograms/— mel spectrogram PNG grid
  plots/       — WER/CER bar chart, MCD comparison
  report/      — evaluation_report.md (human-readable summary)

Metrics: WER · CER (Whisper large-v3) · RTF · MCD(finetuned vs reference)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

EVAL_TEXTS = [
    "sayın yolcularımız, uçuşumuz yaklaşık iki saat sürecektir. emniyet kemerlerinizi bağlayınız.",
    "istanbul'a hoş geldiniz. bagajlarınızı almak için lütfen bant numarasına bakınız.",
    "uçağa biniş kapısı değişmiştir. lütfen yeni kapı numaranızı kontrol ediniz.",
    "değerli yolcularımız, kalkışa hazırlık aşamasındayız. elektronik cihazlarınızı kapatınız.",
    "türk hava yolları olarak hizmetinizde olmaktan memnuniyet duyuyoruz.",
    "hava muhalefeti nedeniyle anlık türbülans yaşanmaktadır. kemerlerinizi bağlı tutunuz.",
    "yolcularımızın dikkatine: uçuşumuz yaklaşık otuz dakika gecikecektir.",
    "kabin ekibimiz kalkış öncesi güvenlik kontrollerini gerçekleştirmektedir.",
    "ankara esenboğa havalimanı'na hoş geldiniz. yerel saat sabah sekizdir.",
    "bagajınızı gözetimsiz bırakmayınız. şüpheli paketleri yetkililere bildiriniz.",
]

SAMPLE_RATE = 22_050
N_FFT       = 1024
HOP_LENGTH  = 256
N_MELS      = 80


# ── Audio generation ───────────────────────────────────────────────────────────

def generate_samples(model, tokenizer, out_dir: Path, label: str, device: str):
    model.eval()
    results = []
    for i, text in enumerate(EVAL_TEXTS):
        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            torch.manual_seed(555)
            t0 = time.time()
            with torch.no_grad():
                out = model(**inputs)
            elapsed = time.time() - t0
            wav = out.waveform.squeeze().cpu().numpy()
            duration = len(wav) / SAMPLE_RATE
            rtf = elapsed / max(duration, 1e-6)
            path = out_dir / f"{label}-{i:02d}.wav"
            sf.write(str(path), wav, samplerate=SAMPLE_RATE)
            results.append({"text": text, "path": str(path),
                            "duration": duration, "rtf": rtf, "wav": wav})
            log.info(f"  [{label}][{i:02d}] {duration:.1f}s | RTF={rtf:.3f}")
        except Exception as e:
            log.warning(f"  Generation failed for sample {i}: {e}")
    return results


# ── ASR metrics ────────────────────────────────────────────────────────────────

def compute_wer_cer(samples, device):
    try:
        import whisper
        from jiwer import cer as j_cer, wer as j_wer
    except ImportError as e:
        log.warning(f"Missing dep for WER: {e}")
        return None, None

    asr = None
    for name, dev in [("large-v3", device), ("small", "cpu")]:
        try:
            asr = whisper.load_model(name, device=dev)
            log.info(f"  ASR: whisper-{name} on {dev}")
            break
        except Exception:
            pass
    if asr is None:
        return None, None

    wers, cers = [], []
    for s in samples:
        try:
            result = asr.transcribe(s["path"], language="tr")
            hyp = result["text"].strip().lower()
            ref = s["text"].lower()
            w, c = j_wer(ref, hyp), j_cer(ref, hyp)
            wers.append(w); cers.append(c)
            log.info(f"  REF: {ref[:65]}")
            log.info(f"  HYP: {hyp[:65]}")
            log.info(f"  WER={w:.2%}  CER={c:.2%}")
        except Exception as e:
            log.warning(f"  Whisper failed for {s['path']}: {e}")

    return (float(np.mean(wers)) if wers else None,
            float(np.mean(cers)) if cers else None)


# ── Mel cepstral distortion ────────────────────────────────────────────────────

def compute_mcd(samples_a, samples_b):
    try:
        import librosa
    except ImportError:
        return None
    mcds = []
    for a, b in zip(samples_a, samples_b):
        try:
            a_w, _ = librosa.load(a["path"], sr=SAMPLE_RATE)
            b_w, _ = librosa.load(b["path"], sr=SAMPLE_RATE)
            n = min(len(a_w), len(b_w))
            am = librosa.feature.mfcc(y=a_w[:n], sr=SAMPLE_RATE, n_mfcc=13)
            bm = librosa.feature.mfcc(y=b_w[:n], sr=SAMPLE_RATE, n_mfcc=13)
            f = min(am.shape[1], bm.shape[1])
            d = am[:, :f] - bm[:, :f]
            mcds.append(float(np.sqrt(2) * np.mean(np.sqrt(np.sum(d ** 2, axis=0)))))
        except Exception as e:
            log.warning(f"  MCD failed: {e}")
    return float(np.mean(mcds)) if mcds else None


# ── MLflow artifact helpers ────────────────────────────────────────────────────

def _log_spectrogram_grid(samples: list, title: str, out_path: Path) -> None:
    """Save mel spectrogram PNG grid for a set of generated samples."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import librosa, librosa.display

        n = min(len(samples), 5)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
        if n == 1:
            axes = [axes]
        for ax, s in zip(axes, samples[:n]):
            wav = s.get("wav")
            if wav is None:
                wav, _ = __import__("librosa").load(s["path"], sr=SAMPLE_RATE)
            mel = librosa.feature.melspectrogram(
                y=wav, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            librosa.display.specshow(
                librosa.power_to_db(mel, ref=np.max),
                ax=ax, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                x_axis="time", y_axis="mel",
            )
            ax.set_title(s["text"][:70], fontsize=8)
            ax.set_ylabel("")
        fig.suptitle(title, fontsize=11, y=1.01)
        plt.tight_layout()
        fig.savefig(str(out_path), dpi=80, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.debug(f"Spectrogram grid failed: {e}")


def _log_comparison_bar(metrics_map: dict, out_path: Path) -> None:
    """Bar chart: WER / CER / RTF per model."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(metrics_map.keys())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric in zip(axes, ("wer", "cer", "rtf")):
            vals  = [metrics_map[l].get(metric) for l in labels]
            colors = ["#e07070", "#4a90d9", "#5cb85c"]
            bars  = ax.bar(labels, [v if v else 0 for v in vals],
                           color=colors[:len(labels)], edgecolor="white")
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
            ax.set_title(metric.upper(), fontsize=11)
            ax.set_ylim(0, max(v for v in vals if v) * 1.25 if any(vals) else 1)
            ax.tick_params(axis="x", labelrotation=15, labelsize=9)
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("3-Way Evaluation: Baseline vs Fine-tuned vs Reference", fontsize=12)
        plt.tight_layout()
        fig.savefig(str(out_path), dpi=80, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        log.debug(f"Comparison bar chart failed: {e}")


def _write_report(metrics_map: dict, mcd_ft_ref, mcd_base_ref, out_path: Path) -> None:
    """Write a human-readable markdown evaluation report."""
    lines = [
        "# MMS-TTS Turkish Fine-tuning — Evaluation Report",
        "",
        "## 3-Way Comparison",
        "",
        "| Model | WER ↓ | CER ↓ | RTF ↓ |",
        "|---|---|---|---|",
    ]
    for label, m in metrics_map.items():
        wer = f"{m['wer']:.1%}" if m.get("wer") is not None else "—"
        cer = f"{m['cer']:.1%}" if m.get("cer") is not None else "—"
        rtf = f"{m['rtf']:.4f}" if m.get("rtf") is not None else "—"
        lines.append(f"| {label} | {wer} | {cer} | {rtf} |")

    lines += ["", "## MCD vs Reference (lower = closer to Turkish reference)"]
    if mcd_base_ref:
        lines.append(f"- Baseline  vs reference: **{mcd_base_ref:.2f} dB**")
    if mcd_ft_ref:
        lines.append(f"- Fine-tuned vs reference: **{mcd_ft_ref:.2f} dB**")
    if mcd_base_ref and mcd_ft_ref:
        delta = mcd_base_ref - mcd_ft_ref
        lines.append(f"- Gap closed: **{delta:+.2f} dB** ({'improvement' if delta > 0 else 'regression'})")

    ft = metrics_map.get("Fine-tuned", {})
    ref = metrics_map.get("Reference", {})
    if ft.get("wer") and ref.get("wer"):
        pct = (1 - ft["wer"] / max(metrics_map.get("Baseline", {}).get("wer", ft["wer"]), 1e-9)) * 100
        lines += [
            "",
            "## Summary",
            f"Fine-tuned WER: **{ft['wer']:.1%}**  |  Reference WER: **{ref['wer']:.1%}**",
            f"WER improvement vs baseline: **{pct:.0f}%**",
        ]

    out_path.write_text("\n".join(lines))


# ── Print summary ──────────────────────────────────────────────────────────────

def print_summary(label: str, metrics: dict):
    log.info("")
    log.info("─" * 55)
    log.info(f"  {label}")
    log.info("─" * 55)
    for k, v in metrics.items():
        if v is not None:
            log.info(f"  {k:<20} {v:.4f}" if isinstance(v, float) else f"  {k:<20} {v}")
    log.info("─" * 55)


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",         type=Path, default=None)
    p.add_argument("--base_model",         default="facebook/mms-tts-eng")
    p.add_argument("--ref_model",          default="facebook/mms-tts-tur")
    p.add_argument("--out_dir",            type=Path, default=Path("/data/tts/eval-outputs"))
    p.add_argument("--hf_cache",           default="/data/tts/hf-cache")
    p.add_argument("--mlflow_uri",         default=os.getenv("MLFLOW_TRACKING_URI", ""))
    p.add_argument("--mlflow_experiment",  default="mms-turkish-tts")
    p.add_argument("--parent_run_id",      default=os.getenv("MLFLOW_PARENT_RUN_ID", ""),
                   help="Link eval runs as nested under the training run")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_cache

    from transformers import VitsModel, VitsTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ── 1. Baseline: mms-tts-eng with ORIGINAL tokenizer ──────────────────────
    log.info(f"Loading baseline: {args.base_model} (original tokenizer)")
    base_tok   = VitsTokenizer.from_pretrained(args.base_model)
    base_model = VitsModel.from_pretrained(args.base_model).to(device)

    baseline_dir = args.out_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    log.info("Generating baseline (English model, Turkish chars mangled)...")
    baseline_samples = generate_samples(base_model, base_tok, baseline_dir, "baseline", device)

    log.info("Evaluating baseline WER/CER...")
    b_wer, b_cer = compute_wer_cer(baseline_samples, device)
    b_rtf = float(np.mean([s["rtf"] for s in baseline_samples])) if baseline_samples else None
    baseline_metrics = {"wer": b_wer, "cer": b_cer, "rtf": b_rtf}
    print_summary("BASELINE — mms-tts-eng (original tokenizer, Turkish chars mangled)", baseline_metrics)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 2. Fine-tuned model ────────────────────────────────────────────────────
    finetuned_metrics, finetuned_samples = {}, []
    if args.checkpoint and args.checkpoint.exists():
        log.info(f"Loading fine-tuned model: {args.checkpoint}")
        ft_tok   = VitsTokenizer.from_pretrained(str(args.checkpoint))
        ft_model = VitsModel.from_pretrained(str(args.checkpoint)).to(device)

        finetuned_dir = args.out_dir / "finetuned"
        finetuned_dir.mkdir(exist_ok=True)
        log.info("Generating fine-tuned samples (full VITS GAN)...")
        finetuned_samples = generate_samples(ft_model, ft_tok, finetuned_dir, "finetuned", device)

        log.info("Evaluating fine-tuned WER/CER...")
        f_wer, f_cer = compute_wer_cer(finetuned_samples, device)
        f_rtf = float(np.mean([s["rtf"] for s in finetuned_samples])) if finetuned_samples else None
        finetuned_metrics = {"wer": f_wer, "cer": f_cer, "rtf": f_rtf}
        print_summary(f"FINE-TUNED — {args.checkpoint.name}", finetuned_metrics)

        del ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 3. Reference: mms-tts-tur ─────────────────────────────────────────────
    ref_metrics, ref_samples = {}, []
    try:
        log.info(f"Loading reference: {args.ref_model}")
        ref_tok   = VitsTokenizer.from_pretrained(args.ref_model)
        ref_model = VitsModel.from_pretrained(args.ref_model).to(device)

        ref_dir = args.out_dir / "reference"
        ref_dir.mkdir(exist_ok=True)
        log.info("Generating reference samples (mms-tts-tur — gold standard)...")
        ref_samples = generate_samples(ref_model, ref_tok, ref_dir, "reference", device)

        log.info("Evaluating reference WER/CER...")
        r_wer, r_cer = compute_wer_cer(ref_samples, device)
        r_rtf = float(np.mean([s["rtf"] for s in ref_samples])) if ref_samples else None
        ref_metrics = {"wer": r_wer, "cer": r_cer, "rtf": r_rtf}
        print_summary("REFERENCE — mms-tts-tur (gold standard)", ref_metrics)

        del ref_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        log.warning(f"Reference model failed: {e}")

    # ── MCD ───────────────────────────────────────────────────────────────────
    mcd_ft_ref   = compute_mcd(finetuned_samples, ref_samples)   if finetuned_samples and ref_samples else None
    mcd_base_ref = compute_mcd(baseline_samples,  ref_samples)   if baseline_samples  and ref_samples else None

    # ── Improvement summary ────────────────────────────────────────────────────
    log.info("")
    log.info("═" * 55)
    log.info("  IMPROVEMENT SUMMARY")
    log.info("═" * 55)
    if b_wer and finetuned_metrics.get("wer"):
        imp = (b_wer - finetuned_metrics["wer"]) / max(b_wer, 1e-9)
        log.info(f"  WER  baseline  → finetuned: {b_wer:.1%} → {finetuned_metrics['wer']:.1%}  ({imp:+.0%})")
    if b_cer and finetuned_metrics.get("cer"):
        imp_c = (b_cer - finetuned_metrics["cer"]) / max(b_cer, 1e-9)
        log.info(f"  CER  baseline  → finetuned: {b_cer:.1%} → {finetuned_metrics['cer']:.1%}  ({imp_c:+.0%})")
    if ref_metrics.get("wer"):
        log.info(f"  WER  reference (mms-tts-tur): {ref_metrics['wer']:.1%}")
    if mcd_base_ref:
        log.info(f"  MCD  baseline  vs reference: {mcd_base_ref:.2f} dB")
    if mcd_ft_ref:
        log.info(f"  MCD  finetuned vs reference: {mcd_ft_ref:.2f} dB")
        if mcd_base_ref:
            log.info(f"       → gap closed by {mcd_base_ref - mcd_ft_ref:.2f} dB")
    log.info("═" * 55)

    # ── Build artifact files ───────────────────────────────────────────────────
    metrics_map = {}
    if baseline_samples:
        metrics_map["Baseline"]  = baseline_metrics
    if finetuned_samples:
        metrics_map["Fine-tuned"] = finetuned_metrics
    if ref_samples:
        metrics_map["Reference"]  = ref_metrics

    # Spectrogram grids
    for label, samples in [("baseline", baseline_samples),
                            ("finetuned", finetuned_samples),
                            ("reference", ref_samples)]:
        if samples:
            _log_spectrogram_grid(
                samples[:5],
                title=f"Mel spectrograms — {label}",
                out_path=plots_dir / f"spectrograms_{label}.png",
            )

    # 3-way bar chart
    if len(metrics_map) >= 2:
        _log_comparison_bar(metrics_map, plots_dir / "comparison_bar.png")

    # Markdown report
    report_path = args.out_dir / "evaluation_report.md"
    _write_report(metrics_map, mcd_ft_ref, mcd_base_ref, report_path)

    # JSON metrics dump (machine-readable)
    json_path = args.out_dir / "metrics.json"
    json_path.write_text(json.dumps({
        "baseline":  baseline_metrics,
        "finetuned": finetuned_metrics,
        "reference": ref_metrics,
        "mcd": {"finetuned_vs_reference": mcd_ft_ref, "baseline_vs_reference": mcd_base_ref},
    }, indent=2, default=lambda x: None if x is None else x))

    # ── MLflow logging (rich artifacts) ───────────────────────────────────────
    if args.mlflow_uri:
        _log_to_mlflow(args, baseline_samples, finetuned_samples, ref_samples,
                       baseline_metrics, finetuned_metrics, ref_metrics,
                       mcd_ft_ref, mcd_base_ref, metrics_map,
                       plots_dir, report_path, json_path, baseline_dir,
                       finetuned_dir if args.checkpoint else None,
                       ref_dir if ref_samples else None)

    log.info(f"Audio + artifacts saved to: {args.out_dir}")


def _log_to_mlflow(
    args, baseline_samples, finetuned_samples, ref_samples,
    baseline_metrics, finetuned_metrics, ref_metrics,
    mcd_ft_ref, mcd_base_ref, metrics_map,
    plots_dir, report_path, json_path,
    baseline_dir, finetuned_dir, ref_dir,
):
    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)

        parent_kwargs = {}
        if args.parent_run_id:
            parent_kwargs = {"run_id": args.parent_run_id}

        def _log_model_run(run_name, model_id, metrics, samples, audio_dir, extra_metrics=None):
            with mlflow.start_run(
                run_name=run_name,
                nested=bool(args.parent_run_id),
                **({} if not args.parent_run_id else {}),
            ) as run:
                mlflow.set_tags({
                    "eval.model":    model_id,
                    "eval.language": "Turkish (tr)",
                    "eval.dataset":  "EVAL_TEXTS (10 airline announcements)",
                    "eval.asr":      "whisper-large-v3",
                })
                mlflow.log_params({"model": model_id, "eval_sentences": len(samples)})
                if metrics.get("wer")  is not None: mlflow.log_metric("eval/wer",  metrics["wer"])
                if metrics.get("cer")  is not None: mlflow.log_metric("eval/cer",  metrics["cer"])
                if metrics.get("rtf")  is not None: mlflow.log_metric("eval/rtf",  metrics["rtf"])
                if extra_metrics:
                    for k, v in extra_metrics.items():
                        if v is not None:
                            mlflow.log_metric(k, v)

                # All WAV artifacts
                if audio_dir and Path(audio_dir).exists():
                    for wav in sorted(Path(audio_dir).glob("*.wav")):
                        mlflow.log_artifact(str(wav), artifact_path="audio")

                # Spectrogram grid
                spec_img = plots_dir / f"spectrograms_{run_name.split('-')[0]}.png"
                if spec_img.exists():
                    mlflow.log_artifact(str(spec_img), artifact_path="spectrograms")

                return run.info.run_id

        # Per-model runs
        _log_model_run("baseline-mms-eng",  args.base_model, baseline_metrics, baseline_samples, baseline_dir)

        if finetuned_samples and finetuned_dir:
            _log_model_run(
                "finetuned-vits-gan", str(args.checkpoint), finetuned_metrics, finetuned_samples, finetuned_dir,
                extra_metrics={
                    "eval/mcd_vs_reference":      mcd_ft_ref,
                    "eval/mcd_baseline_reference": mcd_base_ref,
                },
            )

        if ref_samples and ref_dir:
            _log_model_run("reference-mms-tur", args.ref_model, ref_metrics, ref_samples, ref_dir)

        # Summary run — comparison plots + report (parent or standalone)
        with mlflow.start_run(run_name="eval-summary") as summary_run:
            mlflow.set_tags({
                "eval.type": "3-way comparison",
                "models":    f"{args.base_model} | {args.checkpoint} | {args.ref_model}",
            })
            # Comparison bar chart
            bar = plots_dir / "comparison_bar.png"
            if bar.exists():
                mlflow.log_artifact(str(bar), artifact_path="plots")

            # Markdown report
            if report_path.exists():
                mlflow.log_artifact(str(report_path), artifact_path="report")

            # JSON metrics
            if json_path.exists():
                mlflow.log_artifact(str(json_path), artifact_path="report")

            # Log all flat metrics for easy comparison in MLflow UI
            for label, m in [("baseline", baseline_metrics),
                              ("finetuned", finetuned_metrics),
                              ("reference", ref_metrics)]:
                for k, v in m.items():
                    if v is not None:
                        mlflow.log_metric(f"{label}/{k}", v)
            if mcd_ft_ref:
                mlflow.log_metric("mcd/finetuned_vs_ref", mcd_ft_ref)
            if mcd_base_ref:
                mlflow.log_metric("mcd/baseline_vs_ref", mcd_base_ref)

        log.info(f"Results logged to MLflow: {args.mlflow_uri}")
        log.info(f"  Summary run: {summary_run.info.run_id}")

    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


if __name__ == "__main__":
    main()
