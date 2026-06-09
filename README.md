# TTS Fine-Tuning on OpenShift AI — Turkish Language Adaptation

An example of distributed model fine-tuning using **Kubeflow Trainer v2 (TrainJob)** on Red Hat OpenShift AI. The task — adapting a TTS model to Turkish — is a concrete, measurable illustration of the pattern. The real goal is a reusable reference architecture anyone can apply to their own domain.

---

## Use Case

**Problem:** `facebook/mms-tts-eng` produces English phonetics when given Turkish text. Turkish has sounds not present in English (ş, ğ, ç, ü, ö, ı) and distinct prosody — the result is unintelligible.

**Goal:** Fine-tune the model's text encoder, posterior encoder, and normalizing flow on native Turkish speech data, without touching the HiFi-GAN vocoder (preserving audio quality). Demonstrate measurable improvement in intelligibility and pronunciation, tracked end-to-end in MLflow.

---

## Stack & Why

| Component | Choice | Reason |
|-----------|--------|--------|
| **Base model** | `facebook/mms-tts-eng` (VITS architecture) | Apache-2.0, self-contained TTS — no external vocoder or speaker embeddings needed |
| **Training strategy** | Full GAN (generator + discriminator) | Only fine-tuning the encoder/flow without adversarial loss produces muffled audio; GAN training restores naturalness |
| **Dataset** | `afkfatih/turkish-tts-combined-raw` (~81K samples, 7 speakers) | Largest publicly available Turkish TTS dataset with diverse speakers |
| **Distributed training** | DDP via `torchrun` + Kubeflow Trainer v2 TrainJob | Native multi-node/multi-GPU on Kubernetes with zero boilerplate — `torchrun` reads `PET_*` env vars injected by the Trainer operator |
| **Experiment tracking** | MLflow (OpenShift AI managed instance) | Logs loss curves, audio artifacts, spectrograms, and final model registration in one place |
| **Dependency install** | `initContainer` (`--target /deps` emptyDir) | Keeps the base training image clean; deps are installed once before all trainer containers start |
| **Preprocessing** | `initContainer` (sentinel-guarded) | Data prep runs once per PVC, skipped on resume — training containers start immediately |

---

## Outcome

After fine-tuning on the Turkish dataset:

| Metric | Baseline (`mms-tts-eng`) | Fine-tuned |
|--------|--------------------------|------------|
| WER (Whisper, Turkish) | ~85–95% | Target < 40% |
| CER | ~60–75% | Target < 25% |
| MCD (Mel Cepstral Distortion) | reference | ↓ lower is better |
| Turkish phonemes (ş, ğ, ç…) | absent | present |

Baseline, fine-tuned, and reference (`mms-tts-tur`) audio samples are all logged to MLflow for direct comparison.

---

## Structure

```
examples/tts-finetuning/
├── scripts/
│   ├── preprocess_mms.py     # tokenizer expansion + waveform extraction (run by initContainer)
│   ├── train_vits.py         # full VITS GAN trainer — reads all config from env vars
│   └── evaluate.py           # WER / CER / MCD / RTF — called post-training on rank 0
└── manifests/
    ├── trainjob-tts.yaml     # 2-node TrainJob — the only file you need to edit
    └── kustomization.yaml    # generates tts-mms-scripts ConfigMap from scripts/
```

---

## Deploy

Prerequisites: `smartshop-training` ClusterQueue, `smartshop-shared-storage` PVC, `hf-credentials` secret.

```bash
# Apply ConfigMap + TrainJob in one command
oc apply -k examples/tts-finetuning/

# Watch progress
oc get trainjob mms-turkish-tts -n smartshop -w
oc logs -n smartshop -l training.kubeflow.org/trainjob-name=mms-turkish-tts -f
```

To resume from a checkpoint or cap steps for a smoke test, edit the env vars in `trainjob-tts.yaml`:

```yaml
- name: RESUME_FROM
  value: "/data/tts/checkpoints/mms-turkish/best"
- name: MAX_STEPS
  value: "500"
- name: MAX_TRAIN_SAMPLES
  value: "5000"
```

---

## MLflow

Training logs to the OpenShift AI managed MLflow instance at each eval checkpoint:

- **Metrics:** `mel_loss`, `kl_loss`, `gen_adv`, `disc_loss`, `wer`, `cer`, `mcd`, `rtf`
- **Artifacts:** generated audio (baseline / fine-tuned / reference), spectrograms, loss plots
- **Model registry:** final checkpoint registered as `mms-turkish-tts`
