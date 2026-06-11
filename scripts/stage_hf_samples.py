#!/usr/bin/env python3
"""Stage eval WAVs for Hugging Face — only phrases that pass quality gates."""

import json
import os
import shutil
from pathlib import Path

import soundfile as sf

HF_AUDIO_BASE = (
    "https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio"
)

ALL_PHRASE_IDS = (
    "flight_announce", "welcome", "safety", "farewell", "weather",
    "news_intro", "directions", "question", "tech", "emergency",
)

PHRASE_LABELS = {
    "flight_announce": "Flight announcement",
    "welcome": "Welcome",
    "safety": "Safety briefing",
    "farewell": "Farewell",
    "weather": "Weather",
    "news_intro": "News intro",
    "directions": "Directions",
    "question": "Question",
    "tech": "Technology",
    "emergency": "Emergency",
}


def _wav_duration(path: Path, sr: int = 24_000) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        return 0.0


def _passes_quality(
    phrase_id: str,
    by_id: dict,
    finetuned_wav: Path,
    *,
    min_cer_delta: float,
    min_duration_s: float,
    max_finetuned_cer: float,
) -> tuple[bool, str]:
    if phrase_id not in by_id or "finetuned" not in by_id[phrase_id]:
        return False, "missing finetuned metrics"
    if "baseline" not in by_id[phrase_id]:
        return False, "missing baseline metrics"

    b = by_id[phrase_id]["baseline"]
    f = by_id[phrase_id]["finetuned"]
    b_cer = float(b.get("cer", 1.0))
    f_cer = float(f.get("cer", 1.0))
    cer_delta = b_cer - f_cer

    if cer_delta < min_cer_delta:
        return False, f"ΔCER {cer_delta:.3f} < {min_cer_delta}"
    if f_cer > max_finetuned_cer:
        return False, f"finetuned CER {f_cer:.3f} > {max_finetuned_cer}"
    if not finetuned_wav.exists():
        return False, "finetuned wav missing"

    dur = _wav_duration(finetuned_wav)
    if dur < min_duration_s:
        return False, f"duration {dur:.2f}s < {min_duration_s}s"

    return True, f"ΔCER={cer_delta:.3f} CER={f_cer:.3f} dur={dur:.1f}s"


def _write_audio_section(manifest: dict, out_path: Path, *, finetuned_only: bool):
    lines = [
        "## Audio samples",
        "",
        "**B** = baseline (`unsloth/orpheus-3b-0.1-pretrained`) · **F** = finetuned (this repo).",
        "",
    ]
    for entry in manifest["phrases"]:
        pid = entry["id"]
        ref = entry.get("reference", "")
        f = entry.get("finetuned", {})
        cer = f.get("cer", "?")
        label = entry.get("label", pid)
        cer_s = f"{cer:.2f}" if isinstance(cer, (int, float)) else str(cer)
        lines.append(f"**{label}** — *{ref}* · CER **{cer_s}**")
        lines.append("")
        if finetuned_only:
            lines.append(
                f"<audio controls src=\"{HF_AUDIO_BASE}/{pid}/finetuned.wav\"></audio>"
            )
        else:
            lines.append(
                f"**B** <audio controls src=\"{HF_AUDIO_BASE}/{pid}/baseline.wav\"></audio>"
            )
            lines.append("")
            lines.append(
                f"**F** <audio controls src=\"{HF_AUDIO_BASE}/{pid}/finetuned.wav\"></audio>"
            )
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-cer-delta", type=float,
                   default=float(os.environ.get("HF_MIN_CER_DELTA", "0.05")))
    p.add_argument("--min-duration-s", type=float,
                   default=float(os.environ.get("HF_MIN_DURATION_S", "1.5")))
    p.add_argument("--max-finetuned-cer", type=float,
                   default=float(os.environ.get("HF_MAX_FINETUNED_CER", "0.85")))
    p.add_argument("--write-audio-section",
                   default=os.environ.get("HF_AUDIO_SECTION_PATH", ""))
    p.add_argument("--finetuned-only", action="store_true",
                   default=os.environ.get("HF_FINETUNED_ONLY", "0").lower() in ("1", "true", "yes"))
    args = p.parse_args()

    src = Path(args.eval_dir)
    audio_root = Path(args.out_dir) / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    # Clear prior staged audio (re-run safe)
    if audio_root.exists():
        for child in audio_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    results_path = src / "eval_results.json"
    rows = json.loads(results_path.read_text()) if results_path.exists() else []

    by_id: dict[str, dict] = {}
    for row in rows:
        sid = row.get("sentence", "")
        by_id.setdefault(sid, {})[row.get("model", "")] = row

    manifest = {
        "filter": {
            "min_cer_delta": args.min_cer_delta,
            "min_duration_s": args.min_duration_s,
            "max_finetuned_cer": args.max_finetuned_cer,
            "finetuned_only": args.finetuned_only,
        },
        "phrases": [],
        "excluded": [],
    }

    for phrase_id in ALL_PHRASE_IDS:
        finetuned_src = src / f"finetuned_{phrase_id}.wav"
        baseline_src = src / f"baseline_{phrase_id}.wav"
        if not finetuned_src.exists():
            continue

        ok, reason = _passes_quality(
            phrase_id, by_id, finetuned_src,
            min_cer_delta=args.min_cer_delta,
            min_duration_s=args.min_duration_s,
            max_finetuned_cer=args.max_finetuned_cer,
        )
        if not ok:
            manifest["excluded"].append({"id": phrase_id, "reason": reason})
            print(f"  skip {phrase_id}: {reason}", flush=True)
            continue

        phrase_dir = audio_root / phrase_id
        phrase_dir.mkdir(exist_ok=True)
        shutil.copy2(finetuned_src, phrase_dir / "finetuned.wav")
        if not args.finetuned_only:
            if baseline_src.exists():
                shutil.copy2(baseline_src, phrase_dir / "baseline.wav")
            else:
                print(f"  warn {phrase_id}: baseline wav missing", flush=True)

        spec_f = src / f"finetuned_{phrase_id}_spec.png"
        if spec_f.exists():
            shutil.copy2(spec_f, phrase_dir / "finetuned_spec.png")
        if not args.finetuned_only:
            spec_b = src / f"baseline_{phrase_id}_spec.png"
            if spec_b.exists():
                shutil.copy2(spec_b, phrase_dir / "baseline_spec.png")

        entry = {
            "id": phrase_id,
            "label": PHRASE_LABELS.get(phrase_id, phrase_id),
            "audio": {"finetuned": f"{HF_AUDIO_BASE}/{phrase_id}/finetuned.wav"},
        }
        if not args.finetuned_only:
            entry["audio"]["baseline"] = f"{HF_AUDIO_BASE}/{phrase_id}/baseline.wav"
        if phrase_id in by_id:
            ref_row = by_id[phrase_id].get("baseline") or by_id[phrase_id].get("finetuned") or {}
            entry["reference"] = ref_row.get("text", "")
            if "baseline" in by_id[phrase_id]:
                b = by_id[phrase_id]["baseline"]
                entry["baseline"] = {
                    "wer": b.get("wer"), "cer": b.get("cer"),
                    "rtf": b.get("rtf"), "transcript": b.get("transcript", ""),
                }
            if "finetuned" in by_id[phrase_id]:
                f = by_id[phrase_id]["finetuned"]
                entry["finetuned"] = {
                    "wer": f.get("wer"), "cer": f.get("cer"),
                    "rtf": f.get("rtf"), "transcript": f.get("transcript", ""),
                }
            if entry.get("reference"):
                (phrase_dir / "reference.txt").write_text(entry["reference"], encoding="utf-8")

        manifest["phrases"].append(entry)

    (Path(args.out_dir) / "samples_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (Path(args.out_dir) / "included_phrases.json").write_text(
        json.dumps([p["id"] for p in manifest["phrases"]], indent=2), encoding="utf-8",
    )

    section_path = Path(args.write_audio_section) if args.write_audio_section else None
    if section_path and manifest["phrases"]:
        _write_audio_section(manifest, section_path, finetuned_only=args.finetuned_only)

    print(
        f"Staged {len(manifest['phrases'])}/{len(ALL_PHRASE_IDS)} phrases → {audio_root} "
        f"(excluded {len(manifest['excluded'])})",
        flush=True,
    )


if __name__ == "__main__":
    main()
