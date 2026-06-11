#!/usr/bin/env python3
"""Export MLflow training curves + eval benchmark charts for HF model card."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests

PHRASES = [
    ("flight_announce", "Flight"),
    ("welcome", "Welcome"),
    ("safety", "Safety"),
    ("farewell", "Farewell"),
    ("weather", "Weather"),
    ("news_intro", "News"),
    ("directions", "Directions"),
    ("question", "Question"),
    ("tech", "Tech"),
    ("emergency", "Emergency"),
]


def _fetch_history(uri: str, token: str, run_id: str, key: str) -> list[dict]:
    r = requests.get(
        f"{uri}/api/2.0/mlflow/metrics/get-history",
        params={"run_id": run_id, "metric_key": key, "max_results": 5000},
        headers={"Authorization": f"Bearer {token}", "X-MLflow-Workspace": "smartshop"},
        verify=False,
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("metrics", [])


def _fetch_eval_metrics(uri: str, token: str, run_id: str) -> dict[str, float]:
    r = requests.get(
        f"{uri}/api/2.0/mlflow/runs/get",
        params={"run_id": run_id},
        headers={"Authorization": f"Bearer {token}", "X-MLflow-Workspace": "smartshop"},
        verify=False,
        timeout=60,
    )
    r.raise_for_status()
    return {m["key"]: m["value"] for m in r.json()["run"].get("data", {}).get("metrics", [])}


def _plot_series(out: Path, title: str, series: dict[str, list], ylabel: str, figsize=(10, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    for label, points in series.items():
        if not points:
            continue
        xs = [p["step"] for p in points]
        ys = [p["value"] for p in points]
        ax.plot(xs, ys, label=label, linewidth=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _parse_benchmark(eval_metrics: dict[str, float]) -> dict:
    rows = []
    for pid, label in PHRASES:
        b_wer = eval_metrics.get(f"eval/baseline/{pid}/wer")
        f_wer = eval_metrics.get(f"eval/finetuned/{pid}/wer")
        b_cer = eval_metrics.get(f"eval/baseline/{pid}/cer")
        f_cer = eval_metrics.get(f"eval/finetuned/{pid}/cer")
        if b_wer is None or f_wer is None:
            continue
        rows.append({
            "id": pid, "label": label,
            "baseline_wer": b_wer, "finetuned_wer": f_wer,
            "baseline_cer": b_cer, "finetuned_cer": f_cer,
            "cer_delta": (b_cer or 0) - (f_cer or 0),
            "wer_delta": b_wer - f_wer,
        })
    return {
        "aggregate": {
            "baseline_wer_mean": eval_metrics.get("eval/baseline/wer_mean"),
            "finetuned_wer_mean": eval_metrics.get("eval/finetuned/wer_mean"),
            "baseline_cer_mean": eval_metrics.get("eval/baseline/cer_mean"),
            "finetuned_cer_mean": eval_metrics.get("eval/finetuned/cer_mean"),
            "baseline_rtf_mean": eval_metrics.get("eval/baseline/rtf_mean"),
            "finetuned_rtf_mean": eval_metrics.get("eval/finetuned/rtf_mean"),
            "wer_improvement": eval_metrics.get("eval/wer_improvement"),
            "cer_improvement": eval_metrics.get("eval/cer_improvement"),
        },
        "per_sentence": rows,
    }


def _plot_eval_bars(out: Path, benchmark: dict):
    rows = benchmark["per_sentence"]
    if not rows:
        return
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.bar(x - w / 2, [r["baseline_wer"] for r in rows], w, label="Baseline", color="#94a3b8")
    ax1.bar(x + w / 2, [r["finetuned_wer"] for r in rows], w, label="Finetuned", color="#2563eb")
    ax1.set_title("WER per sentence (lower is better)", fontsize=10)
    ax1.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(x - w / 2, [r["baseline_cer"] for r in rows], w, label="Baseline", color="#94a3b8")
    ax2.bar(x + w / 2, [r["finetuned_cer"] for r in rows], w, label="Finetuned", color="#16a34a")
    ax2.set_title("CER per sentence (lower is better)", fontsize=10)
    ax2.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_cer_improvement(out: Path, benchmark: dict):
    rows = sorted(benchmark["per_sentence"], key=lambda r: r["cer_delta"], reverse=True)
    if not rows:
        return
    labels = [r["label"] for r in rows]
    deltas = [r["cer_delta"] for r in rows]
    colors = ["#16a34a" if d > 0 else "#dc2626" for d in deltas]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color="#64748b", linewidth=0.8)
    ax.set_xlabel("CER improvement (baseline − finetuned)")
    ax.set_title("Per-sentence CER delta — post-train eval (final/)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_aggregate(out: Path, agg: dict):
    if not agg.get("baseline_wer_mean"):
        return
    metrics = ["WER", "CER", "RTF"]
    baseline = [agg["baseline_wer_mean"], agg["baseline_cer_mean"], agg["baseline_rtf_mean"]]
    finetuned = [agg["finetuned_wer_mean"], agg["finetuned_cer_mean"], agg["finetuned_rtf_mean"]]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - w / 2, baseline, w, label="Baseline", color="#94a3b8")
    ax.bar(x + w / 2, finetuned, w, label="Finetuned", color="#2563eb")
    ax.set_xticks(x, metrics)
    ax.set_title("Mean benchmark metrics (10 sentences)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _plot_dashboard(
    out: Path,
    train_loss: list,
    eval_loss: list,
    wer_mean: list,
    cer_mean: list,
    benchmark: dict,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]
    if train_loss:
        ax.plot([p["step"] for p in train_loss], [p["value"] for p in train_loss], label="train/loss", lw=1.2)
    if eval_loss:
        ax.plot([p["step"] for p in eval_loss], [p["value"] for p in eval_loss], label="eval/loss", lw=1.2)
    ax.set_title("Training loss", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if wer_mean:
        ax.plot([p["step"] for p in wer_mean], [p["value"] for p in wer_mean], label="wer/mean", lw=1.2)
    if cer_mean:
        ax.plot([p["step"] for p in cer_mean], [p["value"] for p in cer_mean], label="cer/mean", lw=1.2)
    ax.set_title("In-training WER/CER (4 prompts)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    agg = benchmark.get("aggregate", {})
    if agg.get("baseline_wer_mean"):
        m = ["WER", "CER", "RTF"]
        b = [agg["baseline_wer_mean"], agg["baseline_cer_mean"], agg["baseline_rtf_mean"]]
        f = [agg["finetuned_wer_mean"], agg["finetuned_cer_mean"], agg["finetuned_rtf_mean"]]
        x = np.arange(3)
        ax.bar(x - 0.17, b, 0.34, label="Baseline", color="#94a3b8")
        ax.bar(x + 0.17, f, 0.34, label="Finetuned", color="#2563eb")
        ax.set_xticks(x, m)
    ax.set_title("Post-train means", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1, 1]
    rows = benchmark.get("per_sentence", [])
    if rows:
        labels = [r["label"] for r in rows]
        deltas = [r["cer_delta"] for r in rows]
        colors = ["#16a34a" if d > 0 else "#dc2626" for d in deltas]
        ax.barh(labels, deltas, color=colors)
        ax.axvline(0, color="#64748b", lw=0.8)
    ax.set_title("CER Δ per sentence", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Orpheus Turkish TTS v2 — training & eval overview", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "https://rh-ai.apps.oai-kft-ibm.ibm.rh-ods.com/mlflow")
    token = os.environ.get("MLFLOW_TRACKING_TOKEN") or os.environ.get("MLFLOW_TRACKING_TOKEN".lower())
    train_run = os.environ.get("MLFLOW_TRAIN_RUN_ID", "6804b44335f347849f26da2736aa73df")
    eval_run = os.environ.get("MLFLOW_EVAL_RUN_ID", "27568d2349204af9bf32882f0e8ad1f1")
    out_dir = Path(os.environ.get("OUT_DIR", "eval/mlflow"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if not token:
        raise RuntimeError("Set MLFLOW_TRACKING_TOKEN")

    train_loss = _fetch_history(uri, token, train_run, "train/loss")
    eval_loss = _fetch_history(uri, token, train_run, "eval/loss")
    wer_mean = _fetch_history(uri, token, train_run, "wer/mean")
    cer_mean = _fetch_history(uri, token, train_run, "cer/mean")

    _plot_series(out_dir / "training_loss.png", "Training & eval loss", {
        "train/loss": train_loss, "eval/loss": eval_loss,
    }, "Loss")

    _plot_series(out_dir / "wer_cer_progress.png", "In-training WER/CER mean", {
        "wer/mean": wer_mean, "cer/mean": cer_mean,
    }, "Score (lower is better)")

    per_prompt_wer, per_prompt_cer = {}, {}
    for slug in ("welcome", "flight_announce", "safety", "farewell"):
        per_prompt_wer[f"wer/{slug}"] = _fetch_history(uri, token, train_run, f"wer/{slug}")
        per_prompt_cer[f"cer/{slug}"] = _fetch_history(uri, token, train_run, f"cer/{slug}")
    _plot_series(out_dir / "wer_per_prompt.png", "In-training WER per prompt", per_prompt_wer, "WER")
    _plot_series(out_dir / "cer_per_prompt.png", "In-training CER per prompt", per_prompt_cer, "CER")

    eval_metrics = _fetch_eval_metrics(uri, token, eval_run)
    benchmark = _parse_benchmark(eval_metrics)
    _plot_eval_bars(out_dir / "eval_wer_cer_bars.png", benchmark)
    _plot_cer_improvement(out_dir / "eval_cer_delta.png", benchmark)
    _plot_aggregate(out_dir / "eval_aggregate.png", benchmark["aggregate"])
    _plot_dashboard(out_dir / "dashboard.png", train_loss, eval_loss, wer_mean, cer_mean, benchmark)

    summary = {
        "training_run_id": train_run,
        "eval_run_id": eval_run,
        "mlflow_experiment": "orpheus-turkish-tts",
        "training": {
            "eval_loss": {
                "first": eval_loss[0] if eval_loss else None,
                "last": eval_loss[-1] if eval_loss else None,
                "best": min(eval_loss, key=lambda m: m["value"]) if eval_loss else None,
            },
            "train_loss": {
                "first": train_loss[0] if train_loss else None,
                "last": train_loss[-1] if train_loss else None,
            },
            "wer_mean": {
                "points": len(wer_mean),
                "best": min(wer_mean, key=lambda m: m["value"]) if wer_mean else None,
            },
            "cer_mean": {
                "points": len(cer_mean),
                "best": min(cer_mean, key=lambda m: m["value"]) if cer_mean else None,
            },
        },
        "post_train_eval": benchmark,
    }
    (out_dir / "metrics_export.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
    print(f"Exported MLflow snapshots → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
