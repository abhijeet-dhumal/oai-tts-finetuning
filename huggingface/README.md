---
language: tr
license: apache-2.0
base_model: unsloth/orpheus-3b-0.1-pretrained
tags:
  - text-to-speech
  - tts
  - turkish
  - orpheus
  - snac
  - lora
  - openshift-ai
  - kubeflow-trainer
library_name: transformers
inference: false
---

# Orpheus-3B Turkish TTS (v2)

Fine-tuned [unsloth/orpheus-3b-0.1-pretrained](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) for **Turkish text-to-speech** using LoRA on a SNAC 24 kHz audio-token vocabulary. Trained on Red Hat OpenShift AI with **Kubeflow Trainer v2 (TrainJob)** · [RHOAIENG-62326](https://redhat.atlassian.net/browse/RHOAIENG-62326).

**Use case:** airline-style Turkish announcements where general-purpose TTS lacks Turkish phonology.

| | |
|---|---|
| **Base model** | `unsloth/orpheus-3b-0.1-pretrained` (Llama-3 backbone + SNAC codec tokens) |
| **Fine-tuning** | LoRA r=32, α=64 on attention + MLP projections |
| **Training data** | [afkfatih/turkish-tts-combined-raw](https://huggingface.co/datasets/afkfatih/turkish-tts-combined-raw) — 20,000 samples |
| **Merged weights** | LoRA from `checkpoint-8800` (best eval_loss) |
| **Audio codec** | [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) @ 24 kHz |

## Model hierarchy & specs

```
unsloth/orpheus-3b-0.1-pretrained     # English-biased Orpheus base (frozen)
        └── LoRA adapters (r=32, α=64)  # ~185M trainable params
                └── merged → this repo  # full weights, Turkish TTS
```

| | |
|---|---|
| Architecture | Llama-3 3B causal LM + SNAC audio token head |
| Parameters | ~3B total; ~185M trainable during LoRA fine-tune |
| Vocab | 128,256 text + SNAC codec tokens from offset 128,266 |
| Audio | SNAC 24 kHz · 7 tokens/frame · 3 codebooks |
| Precision | bfloat16 + Flash Attention 2 |
| LoRA | r=32, α=64, dropout 0.05 |
| Target modules | `q/k/v/o_proj`, `gate/up/down_proj` |

Orpheus frames TTS as **causal LM over SNAC audio tokens**: Turkish text → control tokens → autoregressive codec tokens → 24 kHz waveform.

## Dataset specs

| | |
|---|---|
| Source | [afkfatih/turkish-tts-combined-raw](https://huggingface.co/datasets/afkfatih/turkish-tts-combined-raw) |
| Samples used | 20,000 (of ~81k available) |
| Format | WAV + Turkish transcript |
| Preprocessing | Resample 24 kHz mono → SNAC encode → token sequence |
| Seq length | max 4,096 tokens |
| Split | 95% train / 5% eval (from preprocessed pool) |
| Platform | 2× NVIDIA A100 80GB DDP · Kubeflow TrainJob · 8 epochs · 9,504 steps |

## Training stack

| Layer | Component | Version / notes |
|-------|-----------|-----------------|
| Platform | Red Hat OpenShift AI | Managed MLflow, GPU queues (Kueue) |
| Orchestration | Kubeflow Trainer v2 `TrainJob` | `torch-distributed-cuda130-torch210-py312` runtime |
| Distributed | `torchrun` DDP | 2 nodes × 1× NVIDIA A100 80GB |
| Framework | PyTorch 2.10 · Transformers · PEFT | bfloat16 + Flash Attention 2 |
| Codec | [SNAC 24 kHz](https://huggingface.co/hubertsiuzdak/snac_24khz) | 7 tokens/frame · 3 codebooks |
| ASR (eval) | OpenAI Whisper-small | WER/CER via jiwer |
| Tracking | MLflow | Metrics, audio artifacts, traces |
| Storage | PVC `orpheus-tts-storage` | Checkpoints, preprocessed shards, eval WAVs |

## How this model was trained

End-to-end pipeline (reproducible from [GitHub manifests](https://github.com/abhijeet-dhumal/oai-tts-finetuning/tree/main/orpheus-tts)):

| Step | What | Config / script |
|------|------|-----------------|
| 1 | **Preprocess** — resample 24 kHz, SNAC-encode, build token sequences | `train_orpheus.py` · `PREPROCESS_SAMPLES=20000` |
| 2 | **Train** — LoRA fine-tune with audio-only loss masking | `manifests/trainjob-orpheus.yaml` · `scripts/train_orpheus.py` |
| 3 | **Monitor** — eval_loss every 200 steps; audio+WER every 400 steps | MLflow experiment `orpheus-turkish-tts` |
| 4 | **Checkpoint** — save every 800 steps (`save_total_limit=3`) | Best eval_loss @ step **8,800** |
| 5 | **Merge** — LoRA → full weights | `scripts/save_final_orpheus.py` → `final/` |
| 6 | **Evaluate** — 10-sentence benchmark + curated HF upload | `manifests/trainjob-orpheus-eval.yaml` · `scripts/evaluate_orpheus.py` |

### Hyperparameters (v2 run)

| Parameter | Value |
|-----------|-------|
| LoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | 2e-5 |
| Epochs | 8 |
| Global steps | 9,504 |
| Per-device batch | 2 |
| Grad accumulation | 4 |
| **Effective batch** | 2 nodes × 2 GPU × 4 accum = **16** |
| Max sequence length | 4,096 |
| Warmup ratio | 0.1 |
| Precision | bfloat16 + Flash Attention 2 |
| `EVAL_STEPS` | 200 |
| `AUDIO_LOG_STEPS` | 400 |
| `SAVE_STEPS` | 800 |

### Training approach

TTS is **next-token prediction** on interleaved text + SNAC audio tokens. Only audio tokens contribute to loss.

**1. Prompt format** — Orpheus control tokens wrap Turkish text:

```python
ids = tokenizer.encode(text, add_special_tokens=False) + [TOK_EOT]
prompt = [TOK_SOH] + ids + [TOK_EOH, TOK_SOA, TOK_SOS]
# model autoregressively predicts SNAC tokens until TOK_EOA
```

**2. Audio-only loss masking** — prompt positions excluded from cross-entropy:

```python
sos_idx = input_ids.index(TOK_SOS)
labels = [-100] * (sos_idx + 1) + input_ids[sos_idx + 1:]
```

**3. LoRA on frozen backbone:**

```python
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, lora_cfg)
```

**4. Distributed TrainJob** — env-driven config from `trainjob-orpheus.yaml`:

```yaml
# manifests/trainjob-orpheus.yaml (excerpt)
runtimeRef: torch-distributed-cuda130-torch210-py312
numNodes: 2
numProcPerNode: gpu
LEARNING_RATE: "2e-5"
NUM_EPOCHS: "8"
MAX_TRAIN_SAMPLES: "20000"
BATCH_SIZE: "2"
GRAD_ACCUM: "4"
LORA_R: "32"
LORA_ALPHA: "64"
```

### MLflow tracking

| Logged | Detail |
|--------|--------|
| Metrics | `train/loss`, `eval/loss`, `wer/*`, `cer/*`, `rtf/*` per step |
| Artifacts | Step audio WAVs + spectrograms (4 airline prompts during training) |
| Traces | `orpheus_tts_pipeline` → `tokenise` → `model_generate` → `snac_decode` |
| Train run | `6804b44335f347849f26da2736aa73df` |
| Eval-v4 run | `50c177d696164b4a83169292b2109b4e` |

## Results & MLflow

Post-train benchmark (eval-v4): **pretrained base** vs **merged `final/`** · Whisper-small · `temp=0.3, top_p=0.9, rep_penalty=1.15` · finetuned uses length-scaled `min_new_tokens`; baseline capped at `max_new_tokens=400` (English pretrained rarely emits end-of-audio on Turkish).

| Metric | Baseline | Finetuned | Δ |
|--------|----------|-----------|---|
| **WER mean** | 1.576 | **0.723** | −0.854 |
| **CER mean** | 1.224 | **0.410** | −0.814 |
| **RTF mean** | 2.1 | 2.2 | +0.1 |
| **eval_loss** (training) | 9.50 → | **4.35** | @ step 8,800 |

MLflow: `orpheus-turkish-tts` · train `6804b44335f347849f26da2736aa73df` · eval-v4 `50c177d696164b4a83169292b2109b4e`

> **Baseline comparison:** Audio labeled **B** is [`unsloth/orpheus-3b-0.1-pretrained`](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained) (English-pretrained) given the same Turkish text and generation settings. **F** is this checkpoint. Baseline WER/CER vs Turkish references are included for completeness — listen to the pairs below for perceptual comparison.

### MLflow UI (training run)

**Traces** — per-inference span breakdown (`tokenise` → `model_generate` → `snac_decode`) with RTF and duration:

<img src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/mlflow/ui_traces_pipeline.png" width="100%"/>

**Artifacts** — step-indexed WAV + spectrograms browsable in the MLflow UI (example: `flight_announce` @ step 8800):

<img src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/mlflow/ui_artifacts_audio.png" width="100%"/>

### Exported metric charts

<img src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/mlflow/dashboard.png" width="100%"/>

| Chart | Description |
|-------|-------------|
| [ui_traces_pipeline](eval/mlflow/ui_traces_pipeline.png) | MLflow Traces — pipeline spans + RTF |
| [ui_artifacts_audio](eval/mlflow/ui_artifacts_audio.png) | MLflow Artifacts — in-train audio browser |
| [dashboard](eval/mlflow/dashboard.png) | Training loss + in-train WER/CER + eval summary |
| [eval_wer_cer_bars](eval/mlflow/eval_wer_cer_bars.png) | Per-sentence WER/CER |
| [training_loss](eval/mlflow/training_loss.png) | train/eval loss curve |

**Per-sentence benchmark** ([`eval/eval_results.json`](eval/eval_results.json)) — eval-v4:

| Phrase | B-WER | F-WER | B-CER | F-CER | Δ CER |
|--------|-------|-------|-------|-------|-------|
| news_intro | 1.75 | **0.38** | 1.49 | **0.15** | +1.34 |
| directions | 2.00 | **0.50** | 1.56 | **0.22** | +1.33 |
| flight_announce | 1.43 | **0.29** | 0.88 | **0.19** | +0.69 |
| farewell | 1.00 | **0.60** | 0.78 | **0.29** | +0.49 |
| welcome | 3.33 | **0.67** | 2.79 | **0.13** | +2.67 |
| weather | 1.00 | **0.86** | 0.89 | **0.28** | +0.62 |
| tech | 1.00 | **0.67** | 0.90 | **0.43** | +0.47 |
| emergency | 1.00 | **0.86** | 0.83 | **0.61** | +0.22 |
| question | 2.25 | **0.75** | 1.24 | **0.51** | +0.73 |
| safety | — | — | — | — | excluded (ΔCER < 0) |

## Audio samples

9 curated phrases (ΔCER ≥ 5pp, finetuned CER ≤ 0.85, duration ≥ 1.5s). Excluded: **safety** only. **B** = baseline (`unsloth/orpheus-3b-0.1-pretrained`, English-only) · **F** = finetuned. Baseline clips use capped generation (`max_new_tokens=400`) because the pretrained model rarely emits end-of-audio on OOD Turkish and would otherwise ramble for 10–15s of unrelated speech.

**Welcome** — *istanbul'a hoş geldiniz.* · F-CER **0.13**

**B** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/welcome/baseline.wav"></audio>

**F** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/welcome/finetuned.wav"></audio>

**Flight** — *sayın yolcularımız, uçuşumuz yaklaşık iki saat sürecektir.* · F-CER **0.19**

**B** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/flight_announce/baseline.wav"></audio>

**F** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/flight_announce/finetuned.wav"></audio>

**Directions** — *düz gidin, sonra sağa dönün ve köprüyü geçin.* · F-CER **0.22**

**B** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/directions/baseline.wav"></audio>

**F** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/directions/finetuned.wav"></audio>

**News intro** — *ana haberlere geçmeden önce önemli bir duyurumuz var.* · F-CER **0.15**

**B** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/news_intro/baseline.wav"></audio>

**F** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/news_intro/finetuned.wav"></audio>

**Farewell** — *teşekkür ederiz, iyi yolculuklar dileriz.* · F-CER **0.29**

**B** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/farewell/baseline.wav"></audio>

**F** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/farewell/finetuned.wav"></audio>

**Weather** — *bugün istanbul'da hava bulutlu ve serin olacak.* · F-CER **0.28**

**B** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/weather/baseline.wav"></audio>

**F** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/weather/finetuned.wav"></audio>

**Question** — *yarın toplantıya katılabilir misiniz?* · F-CER **0.51**

**B** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/question/baseline.wav"></audio>

**F** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/question/finetuned.wav"></audio>

**Tech** — *yapay zeka teknolojisi her geçen gün gelişmeye devam ediyor.* · F-CER **0.43**

**B** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/tech/baseline.wav"></audio>

**F** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/tech/finetuned.wav"></audio>

**Emergency** — *acil durum! lütfen binayı derhal tahliye edin.* · F-CER **0.61**

**B** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/emergency/baseline.wav"></audio>

**F** <audio controls src="https://huggingface.co/AbDhumal/orpheus-3b-turkish-tts-v2/resolve/main/eval/audio/emergency/finetuned.wav"></audio>

## Quick Start

```bash
pip install torch transformers peft soundfile librosa snac
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

MODEL = "AbDhumal/orpheus-3b-turkish-tts-v2"
V = 128_256
TOK_SOH, TOK_EOH, TOK_SOA, TOK_SOS, TOK_EOA, TOK_EOT = V+3, V+4, V+5, V+1, V+6, V+9

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).eval().cuda()
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

text = "istanbul'a hoş geldiniz."
ids = tokenizer.encode(text, add_special_tokens=False) + [TOK_EOT]
prompt = [TOK_SOH] + ids + [TOK_EOH, TOK_SOA, TOK_SOS]
min_new = max(80, len(text) * 7)  # match evaluate_orpheus.py
out = model.generate(torch.tensor([prompt]).cuda(), max_new_tokens=1500, min_new_tokens=min_new,
    do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.15, eos_token_id=TOK_EOA)
# SNAC decode → 24 kHz WAV (see scripts/evaluate_orpheus.py)
```

## Reproduction

```bash
git clone https://github.com/abhijeet-dhumal/oai-tts-finetuning.git
cd oai-tts-finetuning/orpheus-tts

# ConfigMap (scripts) + PVC
oc kustomize . | oc apply -f - -n <namespace>

# Train (v2 hyperparameters)
oc apply -f manifests/trainjob-orpheus.yaml

# After training — merge best LoRA checkpoint
python scripts/save_final_orpheus.py   # or run on cluster PVC

# Post-train eval
oc apply -f manifests/trainjob-orpheus-eval.yaml
```

| File | Purpose |
|------|---------|
| `manifests/trainjob-orpheus.yaml` | 2-node TrainJob — all hyperparameters as env vars |
| `manifests/trainjob-orpheus-eval.yaml` | Post-train WER/CER + audio benchmark |
| `manifests/job-hf-upload.yaml` | HF publish (`UPLOAD_MODE=eval\|full\|readme`) |
| `scripts/train_orpheus.py` | Preprocess + LoRA training loop |
| `scripts/evaluate_orpheus.py` | Eval harness (Whisper WER/CER, SNAC decode) |
| `scripts/save_final_orpheus.py` | Merge LoRA → `final/` weights |
| `scripts/stage_hf_samples.py` | Curated HF audio staging (quality gates) |
| `kustomization.yaml` | PVC + `orpheus-tts-scripts` ConfigMap |

| Artifact | Link |
|----------|------|
| Eval metrics | [`eval/eval_results.json`](eval/eval_results.json) |
| Curated audio index | [`eval/samples_manifest.json`](eval/samples_manifest.json) |
| MLflow exports | [`eval/mlflow/`](eval/mlflow/) |

**Limitations:** PoC · English base cannot speak Turkish · Whisper metrics noisy · Not production-certified.

**License:** [unsloth/orpheus-3b-0.1-pretrained](https://huggingface.co/unsloth/orpheus-3b-0.1-pretrained)
