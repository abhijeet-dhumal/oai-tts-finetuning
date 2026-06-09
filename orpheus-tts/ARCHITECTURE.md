# Architecture — Orpheus TTS Fine-Tuning on Red Hat OpenShift AI

## Overview

This example demonstrates distributed LLM-based TTS fine-tuning on **Red Hat OpenShift AI** using the **Kubeflow Trainer v2 TrainJob** primitive. The workload adapts Orpheus-3B — a codec language model that generates speech as discrete audio tokens — to Turkish, using standard next-token prediction training.

The pattern is general-purpose: the same manifest and script template applies to any LLM-based TTS fine-tuning workload. Swap the base model, dataset, and target language without changing the infrastructure.

---

## Use Case

`unsloth/orpheus-3b-0.1-pretrained` is an open-weight 3B-parameter model built on Llama-3 that generates speech by producing SNAC codec tokens rather than waveform samples directly. It has no Turkish phonology by default — given Turkish text it produces incoherent output. The goal is to fine-tune it on native Turkish text-audio pairs, re-encoded with the correct Llama-3 tokenizer vocabulary.

---

## Model Architecture

Orpheus-3B is a **codec language model**: it treats TTS as a sequence-to-sequence task where the output is a stream of discrete audio codec tokens rather than a mel-spectrogram or waveform.

```
Turkish text
      │
      ▼
 Llama-3 Tokenizer      ← expands text to BPE tokens
      │
      ▼
 Orpheus-3B (Llama-3 backbone, 3.3B params)
   · 32 transformer layers
   · 4096 hidden dim, 32 attention heads
   · Extended vocabulary: 128,256 (Llama-3 base) + 10 special tokens + 18,432 SNAC codes
      │
      ▼
 Token stream: [SOH] text_tokens [EOT][EOH][SOA][SOS] audio_tokens [EOA]
      │
      ▼
 SNAC decoder (hubertsiuzdak/snac_24khz)
   · 7 interleaved tokens/frame across 3 codebooks (L0/L1/L2)
   · Reconstructs 24kHz waveform from codec codes
      │
      ▼
 Audio output @ 24kHz
```

### Token sequence format

Each training sample is a single token sequence encoding both text and audio:

```
[SOH] <text BPE tokens> [EOT][EOH][SOA][SOS]
  <L0_0> <L1_0> <L2_0> <L2_1> <L1_1> <L2_2> <L2_3>   ← frame 0
  <L0_1> <L1_2> <L2_4> <L2_5> <L1_3> <L2_6> <L2_7>   ← frame 1
  ...
[EOA]
```

Codes are offset by `LLAMA_VOCAB + 10 = 128,266` to place them outside the text vocabulary. L0 offset = 128,266, L1 = +4096, L2 = +8192.

This format matches [canopylabs/orpheus-tts](https://github.com/canopyai/Orpheus-TTS) exactly. Using a Llama-2-based vocabulary (64K) for a Llama-3 model (128K) was the root cause of the original "gibberish English" output.

---

## Training Pipeline

Training is standard **causal language modelling** — cross-entropy loss over the audio token positions only. HuggingFace `Trainer` handles DDP, gradient accumulation, checkpointing, and LR scheduling transparently.

### Loss function

Cross-entropy over predicted vs. ground-truth SNAC tokens at each audio position. Text token positions use `labels = -100` (masked, no loss contribution). This forces the model to learn audio generation conditioned on the input text.

### Optimiser

AdamW, `lr=1e-4`, cosine decay, warmup ratio 0.05. `torch.bfloat16` for all model weights and activations (A100 native).

### Effective batch size

`batch_size=2 × grad_accum=4 × nodes=2 = 16` sequences per gradient update.

---

## Data Pipeline

### Raw dataset

`afkfatih/turkish-tts-combined-raw` — ~81,000 text+audio pairs, native Turkish studio recordings.

### Preprocessing (`preprocess()` in `train_orpheus.py`)

Runs **before** `torchrun` initialises DDP, distributed across all nodes using `PET_NODE_RANK` / `PET_NNODES`:

```
Node 0                              Node 1
  Load raw dataset                    Load raw dataset
  Select samples 0–4999               Select samples 5000–9999
  For each sample:                    For each sample:
    decode audio (soundfile)            decode audio (soundfile)
    resample to 24kHz (scipy)           resample to 24kHz (scipy)
    SNAC encode → L0/L1/L2 codes       SNAC encode → L0/L1/L2 codes
    tokenize text (Llama-3 BPE)        tokenize text (Llama-3 BPE)
    interleave → token sequence        interleave → token sequence
  Save shard_0/ to PVC                Save shard_1/ to PVC
  Touch .shard_0.done                 Touch .shard_1.done
  Wait for .shard_1.done
  Merge shards → preprocessed/
  Touch .done
```

Both nodes then wait for `.done` before `torchrun` starts. On restart the sentinel is detected and preprocessing is skipped entirely.

---

## Distributed Training Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Red Hat OpenShift AI                                                │
│                                                                      │
│  ┌────────────────────────────┐   ┌────────────────────────────┐     │
│  │  TrainJob Pod node-0       │   │  TrainJob Pod node-1       │     │
│  │  1× NVIDIA A100-80GB       │   │  1× NVIDIA A100-80GB       │     │
│  │                            │   │                            │     │
│  │  [preprocess] shard 0–4999 │   │  [preprocess] shard 5000+  │     │
│  │  torchrun rank 0           │◄──►  torchrun rank 1           │ NCCL│
│  │    train_orpheus.py        │   │    train_orpheus.py        │     │
│  │    (logs + MLflow)         │   │    (silent DDP worker)     │     │
│  └────────────┬───────────────┘   └────────────┬───────────────┘     │
│               └──────────────┬─────────────────┘                    │
│                              │ shared PVC (orpheus-tts-storage 150Gi)│
│               ┌──────────────▼──────────────┐                       │
│               │  /data/orpheus/              │                       │
│               │  ├─ hf-cache/               │  model + SNAC weights │
│               │  ├─ deps/                   │  pip-installed pkgs   │
│               │  └─ checkpoints/            │                       │
│               │      ├─ preprocessed/       │  Arrow dataset        │
│               │      ├─ turkish/            │  HF Trainer ckpts     │
│               │      └─ turkish/final/      │  safetensors weights  │
│               └─────────────────────────────┘                       │
│                                                                      │
│  Kubeflow Trainer v2 operator                                        │
│  · Provisions pods + headless service for DNS rendezvous             │
│  · Injects PET_MASTER_ADDR / PET_NODE_RANK / PET_NNODES —           │
│    consumed natively by torchrun                                     │
│  · Kueue ClusterQueue enforces GPU quota per namespace               │
│                                                                      │
│  MLflow (managed instance)                                           │
│  · Metrics per step: loss, perplexity, wer/*, cer/*, rtf/*          │
│  · Audio + spectrogram artifacts every AUDIO_LOG_STEPS              │
│  · Traces: tokenise → model_generate → snac_decode per inference    │
│  · Model Registry: checkpoint version per SAVE_STEPS                │
└──────────────────────────────────────────────────────────────────────┘
```

### Kubeflow Trainer v2 — TrainJob

TrainJob handles multi-node pod provisioning, DNS-based rendezvous, and `PET_*` environment injection. `torchrun` reads these natively — no custom launcher wrapper. Kueue enforces GPU quota. Scripts are served from a **ConfigMap** built by Kustomize — updated with a single `kubectl kustomize . | oc apply -f -` without rebuilding any container image.

---

## MLflow Integration

### Metrics (step-indexed)
- `loss`, `eval_loss`, `perplexity`, `eval_perplexity`
- `wer/<prompt>`, `cer/<prompt>` — Whisper-small on CPU, language=`tr`
- `rtf/<prompt>`, `rtf/mean` — real-time factor per eval sentence
- `rtf_latest` — logged inline per inference call

### Artifacts
```
audio/
  step_000000/pretrained_baseline/<prompt>/
    <prompt>.wav
    <prompt>_spec.png
  step_000050/finetuned/<prompt>/
    <prompt>.wav
    <prompt>_spec.png
  ...
  final/<prompt>/
checkpoints/
  step_000200/
    config.json
    tokenizer_config.json
```

### Traces
Each `generate_audio()` call produces a nested trace:
```
orpheus_tts_pipeline  [CHAIN]  inputs: {text, training_step}
  ├── tokenise        [PARSER]   outputs: {n_tokens}
  ├── model_generate  [LLM]      outputs: {generated_tokens}
  └── snac_decode     [RETRIEVER] outputs: {duration_s, sample_rate, samples}
                                 outputs: {duration_s, rtf, elapsed_s}
```

### Model Registry
`orpheus-turkish-tts` — a new version registered at each `SAVE_STEPS` checkpoint with tags:
- `training_step`: gradient step number
- `checkpoint_pvc_path`: full path to weights on PVC
- `stage: final` on the last version

---

## Evaluation Sentences

Five fixed Turkish prompts are used at every eval interval, covering TTS production use cases:

| Label | Turkish | English |
|-------|---------|---------|
| `welcome` | İstanbul'a hoş geldiniz. Uçuşunuz için teşekkür ederiz. | Welcome to Istanbul... |
| `safety` | Güvenlik nedeniyle elektronik cihazlarınızı kapalı tutunuz. | Please keep electronic devices off... |
| `flight_announce` | Sayın yolcularımız, uçağımız kalkışa hazırdır. | Dear passengers, our aircraft is ready for takeoff. |
| `farewell` | Teşekkür ederiz, iyi yolculuklar dileriz. | Thank you, have a good journey. |

---

## References

1. Lacombe, Y. & Kumar, A. — "Orpheus-TTS: High Quality Open Source TTS," canopyai, 2025.
   [https://github.com/canopyai/Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)

2. Siuzdak, H. — "SNAC: Multi-Scale Neural Audio Codec," 2024.
   [https://github.com/hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)

3. Dubey, A. et al. — "The Llama 3 Herd of Models," Meta AI, 2024.
   [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)

4. Radford, A. et al. — "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), OpenAI, 2022.
   [https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)

5. Kubeflow Trainer v2 — TrainJob API reference.
   [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)

6. Wolf, T. et al. — "HuggingFace Transformers: State-of-the-Art NLP," EMNLP 2020.
   [https://arxiv.org/abs/1910.03771](https://arxiv.org/abs/1910.03771)
