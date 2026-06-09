"""
Full VITS GAN fine-tuning: facebook/mms-tts-eng → Turkish language adaptation.

Training pipeline (community-standard ylacombe approach, self-contained):
  1. Posterior encoder:  linear_spec(real_audio)   → z_q, m_q, logs_q
  2. Text encoder:       input_ids                 → m_p, logs_p, hidden
  3. Flow (forward):     z_q                       → z_p  (prior space)
  4. MAS alignment:      z_p, m_p, logs_p          → monotonic text↔audio map
  5. KL loss:            KL(posterior || aligned_prior)
  6. Decode (teacher):   z_q → fake_wav            (bypasses duration predictor)
  7. Mel loss:           L1(mel(fake_wav), mel(real_wav))
  8. GAN:                discriminator(real) vs discriminator(fake)

Trained:   text_encoder · posterior_encoder · flow
Frozen:    decoder/vocoder (HiFi-GAN) — preserves audio synthesis quality, saves ~50% VRAM
Disc init: random (standard VITS approach — no pretrained discriminator available)

Loss weights (original VITS paper, Kim et al. ICML 2021, c_mel=45, c_kl=1.0):
  mel×45  +  KL×1.0  +  gen_adv×1  +  feature_maps×1

Distributed: torchrun (DDP) — Kubeflow Trainer v2 injects PET_* env vars natively
MLflow:      auto-integrated via MLFLOW_TRACKING_URI env var

All CLI args fall back to env vars so the command can be simply:
  torchrun /scripts/train_vits.py [--run_eval]
"""

import argparse
import gc
import logging
import math
import os
import sys
import time
import warnings
import urllib3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

TARGET_SR   = 16_000  # MMS-TTS native sample rate (config.sampling_rate=16000)
N_FFT       = 1024
HOP_LENGTH  = 256
WIN_LENGTH  = 1024
N_MELS      = 80
SPEC_BINS   = N_FFT // 2 + 1  # 513  — posterior encoder input channels

EVAL_TEXTS = [
    "sayın yolcularımız, uçuşumuz yaklaşık iki saat sürecektir.",
    "istanbul'a hoş geldiniz. bagajlarınızı almak için lütfen bant numarasına bakınız.",
    "uçağa biniş kapısı değişmiştir. lütfen yeni kapı numaranızı kontrol ediniz.",
    "güvenlik nedeniyle elektronik cihazlarınızı kapalı tutunuz.",
    "teşekkür ederiz, iyi yolculuklar dileriz.",
]


# ── Spectral transforms (built on PyTorch STFT, no extra deps) ─────────────────


def compute_linear_spec(wav: torch.Tensor) -> torch.Tensor:
    """Batch linear spectrogram for posterior encoder.  (B,T) → (B, 513, F).

    Uses a single batched torch.stft call — no Python loop over batch items.
    All waveforms in a batch are padded to the same length by the collator so
    the resulting spectrogram tensors are identically shaped.
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    window = torch.hann_window(WIN_LENGTH, device=wav.device)
    spec = torch.stft(
        wav.reshape(-1, wav.shape[-1]),           # (B, T) treated as (B*1, T)
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=window, return_complex=True,
    ).abs()                                        # (B, SPEC_BINS, F)
    return spec


def _build_mel_filterbank(n_fft: int, n_mels: int, sr: int, device: torch.device) -> torch.Tensor:
    """HTK-style mel filterbank as a pure-torch fallback."""
    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_f = n_fft // 2 + 1
    f_min, f_max = 0.0, sr / 2.0
    mel_min, mel_max = hz_to_mel(f_min), hz_to_mel(f_max)
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts  = torch.tensor([mel_to_hz(m.item()) for m in mel_pts])
    bin_pts = ((n_fft + 1) * hz_pts / sr).long().clamp(0, n_f - 1)

    fb = torch.zeros(n_mels, n_f, device=device)
    for m in range(n_mels):
        lo, pk, hi = bin_pts[m].item(), bin_pts[m + 1].item(), bin_pts[m + 2].item()
        for f in range(lo, pk + 1):
            if pk > lo:
                fb[m, f] = (f - lo) / (pk - lo)
        for f in range(pk, hi + 1):
            if hi > pk:
                fb[m, f] = (hi - f) / (hi - pk)
    return fb


def compute_mel(wav: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Log-mel spectrogram for mel loss.  (B,T) → (B, 80, F)."""
    try:
        import torchaudio.transforms as T
        mel_fn = T.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH, n_mels=N_MELS, power=1.0,
            norm="slaney", mel_scale="htk",
        ).to(device)
        mel = mel_fn(wav)
    except Exception as e:
        log.debug(f"torchaudio MelSpectrogram failed ({e}), using pure-torch filterbank")
        spec = compute_linear_spec(wav)          # (B, n_fft//2+1, T)
        fb   = _build_mel_filterbank(N_FFT, N_MELS, TARGET_SR, device)
        mel  = torch.einsum("mf,bft->bmt", fb, spec)
    return torch.log(mel.clamp(min=1e-5))


# ── VITS Discriminator (MPD + MSD, community standard) ────────────────────────

class _MPDSub(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.period = period
        k, s = 5, 3
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(1,    32,   (k, 1), (s, 1), (k // 2, 0))),
            nn.utils.weight_norm(nn.Conv2d(32,   128,  (k, 1), (s, 1), (k // 2, 0))),
            nn.utils.weight_norm(nn.Conv2d(128,  512,  (k, 1), (s, 1), (k // 2, 0))),
            nn.utils.weight_norm(nn.Conv2d(512,  1024, (k, 1), (s, 1), (k // 2, 0))),
            nn.utils.weight_norm(nn.Conv2d(1024, 1024, (k, 1), 1,      (k // 2, 0))),
        ])
        self.post = nn.utils.weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        pad = (self.period - T % self.period) % self.period
        x = F.pad(x, (0, pad), mode="reflect")
        x = x.view(B, 1, x.shape[-1] // self.period, self.period)
        fmaps = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class _MSDSub(nn.Module):
    def __init__(self, use_sn: bool = False):
        super().__init__()
        norm = nn.utils.spectral_norm if use_sn else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1,    16,   15, 1,  padding=7)),
            norm(nn.Conv1d(16,   64,   41, 4,  groups=4,   padding=20)),
            norm(nn.Conv1d(64,   256,  41, 4,  groups=16,  padding=20)),
            norm(nn.Conv1d(256,  1024, 41, 4,  groups=64,  padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 4,  groups=256, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5,  1,  padding=2)),
        ])
        self.post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)          # (B,1,T)
        fmaps = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmaps.append(x)
        x = self.post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class VitsDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = nn.ModuleList([_MPDSub(p) for p in (2, 3, 5, 7, 11)])
        self.msd = nn.ModuleList([_MSDSub(use_sn=(i == 0)) for i in range(3)])
        self.msd_pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        r_scores, f_scores, r_fmaps, f_fmaps = [], [], [], []
        for disc in self.mpd:
            rs, rf = disc(real)
            fs, ff = disc(fake)
            r_scores.append(rs); f_scores.append(fs)
            r_fmaps.append(rf);  f_fmaps.append(ff)
        x_r, x_f = real, fake
        for disc, pool in zip(self.msd, self.msd_pools):
            x_r = pool(x_r); x_f = pool(x_f)
            rs, rf = disc(x_r)
            fs, ff = disc(x_f)
            r_scores.append(rs); f_scores.append(fs)
            r_fmaps.append(rf);  f_fmaps.append(ff)
        return r_scores, f_scores, r_fmaps, f_fmaps


# ── Loss functions ─────────────────────────────────────────────────────────────

def disc_loss(real_scores: List, fake_scores: List) -> torch.Tensor:
    loss = real_scores[0].new_zeros(1)
    for r, f in zip(real_scores, fake_scores):
        loss = loss + torch.mean((1 - r) ** 2) + torch.mean(f ** 2)
    return loss


def gen_adv_loss(fake_scores: List) -> torch.Tensor:
    loss = fake_scores[0].new_zeros(1)
    for f in fake_scores:
        loss = loss + torch.mean((1 - f) ** 2)
    return loss


def fmap_loss(real_fmaps: List, fake_fmaps: List) -> torch.Tensor:
    loss = real_fmaps[0][0].new_zeros(1)
    for rf_list, ff_list in zip(real_fmaps, fake_fmaps):
        for rf, ff in zip(rf_list, ff_list):
            # Trim trailing time dimension to minimum — real/fake can differ by a few frames
            t = min(rf.shape[-1], ff.shape[-1])
            loss = loss + torch.mean(torch.abs(rf[..., :t].detach() - ff[..., :t]))
    return loss * 2


def kl_loss(
    z_p: torch.Tensor, logs_q: torch.Tensor,
    m_p: torch.Tensor, logs_p: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """ELBO KL term: KL(q(z|x) || p(z|c))."""
    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * (z_p - m_p).pow(2) * torch.exp(-2.0 * logs_p)
    return (kl * mask).mean()


def mel_loss(
    real_wav: torch.Tensor, fake_wav: torch.Tensor, device: torch.device
) -> torch.Tensor:
    mel_r = compute_mel(real_wav, device)
    mel_f = compute_mel(fake_wav, device)
    T = min(mel_r.shape[-1], mel_f.shape[-1])
    return F.l1_loss(mel_f[..., :T], mel_r[..., :T])


# ── Monotonic Alignment Search (pure PyTorch, no Cython) ──────────────────────

def monotonic_alignment_search(
    attn_logprob: torch.Tensor,
    text_lengths: torch.Tensor,
    spec_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Viterbi-style MAS.  Uses numpy for the inner DP loops — ~100× faster
    than torch element-wise access on CPU/GPU with Python overhead.

    attn_logprob: (B, T_text, T_spec)  — log-likelihood matrix
    Returns: path  (B, T_text, T_spec)  — one-hot assignment (float)
    """
    np_lp  = attn_logprob.detach().cpu().float().numpy()  # (B, T_text, T_spec)
    B, T_text, T_spec = np_lp.shape
    out    = np.zeros((B, T_text, T_spec), dtype=np.float32)

    for b in range(B):
        S   = int(text_lengths[b])
        T   = int(spec_lengths[b])

        # Monotonic alignment requires at least one spec frame per text token.
        # When T < S the batch sample is pathological (long text, very short audio).
        # Raising RuntimeError lets the training loop's skip-batch handler discard
        # the entire batch cleanly (with gradient zeroing).
        if T < S or S == 0 or T == 0:
            raise RuntimeError(
                f"MAS impossible for batch item {b}: T={T} spec frames < S={S} text tokens. "
                "Batch will be skipped."
            )

        lp  = np_lp[b, :S, :T]                   # (S, T)

        # ── Forward DP ──────────────────────────────────────────────────────
        Q   = np.full((S, T), -1e9, dtype=np.float32)
        Q[0, 0] = lp[0, 0]
        for s in range(1, S):
            t_lo = s
            t_hi = T - (S - s - 1)
            # vectorised update across t for this row
            prev_row  = Q[s - 1, t_lo - 1 : t_hi - 1]   # Q[s-1, t-1]
            curr_prev = Q[s,     t_lo - 1 : t_hi - 1]   # Q[s,   t-1]
            Q[s, t_lo:t_hi] = np.maximum(prev_row, curr_prev) + lp[s, t_lo:t_hi]

        # ── Backtrack ────────────────────────────────────────────────────────
        t = T - 1
        for s in range(S - 1, -1, -1):
            out[b, s, t] = 1.0
            if s > 0 and t > 0 and Q[s - 1, t - 1] >= Q[s, t - 1]:
                t -= 1

    return torch.from_numpy(out).to(attn_logprob.device)


# ── Data collator ──────────────────────────────────────────────────────────────

class VitsGANCollator:
    """Pads input_ids and waveforms; returns lengths for masking."""

    def __init__(self, pad_id: int = 0, max_wav_len: int = TARGET_SR * 8):
        self.pad_id = pad_id
        self.max_wav_len = max_wav_len

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        ids  = [torch.tensor(f["input_ids"],  dtype=torch.long)  for f in features]
        wavs = [torch.tensor(f["waveform"],   dtype=torch.float32)[:self.max_wav_len]
                for f in features]

        id_lens  = torch.tensor([len(x) for x in ids],  dtype=torch.long)
        wav_lens = torch.tensor([len(w) for w in wavs], dtype=torch.long)

        ids_pad  = torch.zeros(len(ids),  max(len(x) for x in ids),  dtype=torch.long)
        wavs_pad = torch.zeros(len(wavs), max(len(w) for w in wavs), dtype=torch.float32)
        attn     = torch.zeros(len(ids),  ids_pad.shape[1],           dtype=torch.float32)

        for i, (x, w) in enumerate(zip(ids, wavs)):
            ids_pad[i, :len(x)]  = x
            wavs_pad[i, :len(w)] = w
            attn[i, :len(x)]     = 1.0

        return {
            "input_ids":       ids_pad,   # (B, T_text)
            "attention_mask":  attn,       # (B, T_text)
            "waveform":        wavs_pad,   # (B, T_wav)
            "text_lengths":    id_lens,    # (B,)
            "wav_lengths":     wav_lens,   # (B,)
        }


# ── Trainer ────────────────────────────────────────────────────────────────────

class VitsGANTrainer:

    def __init__(self, model, tokenizer, discriminator, args, mlflow_run=None):
        self.model       = model
        self.tokenizer   = tokenizer
        self.disc        = discriminator
        self.args        = args
        self.mlflow_run  = mlflow_run
        self._setup_trainable()

    def _setup_trainable(self):
        # Freeze decoder (HiFi-GAN) — preserves synthesis quality, saves 50% VRAM
        for p in self.model.decoder.parameters():
            p.requires_grad_(False)
        # Train everything else: text_encoder, posterior_encoder, flow
        for name, p in self.model.named_parameters():
            if "decoder" not in name:
                p.requires_grad_(True)

        # Small weight init — prevents discriminator from overflowing at step 1
        # (random init at default scale causes NaN gradients with LR=2e-4)
        for m in self.disc.modules():
            if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Linear)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        gen_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        gen_total     = sum(p.numel() for p in self.model.parameters())
        disc_params   = sum(p.numel() for p in self.disc.parameters())
        log.info(f"Generator trainable: {gen_trainable:,} / {gen_total:,} "
                 f"({100*gen_trainable/gen_total:.1f}%)")
        log.info(f"Discriminator params: {disc_params:,}")
        log.info("Frozen: model.decoder (HiFi-GAN vocoder)")

    # ── Training step ──────────────────────────────────────────────────────────

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        device: torch.device,
        scaler_g,
        scaler_d,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
        accelerator,
    ) -> Dict[str, float]:

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        waveform       = batch["waveform"].to(device)
        text_lengths   = batch["text_lengths"].to(device)
        wav_lengths    = batch["wav_lengths"].to(device)

        use_fp16 = scaler_g is not None and scaler_g.is_enabled()
        # ── Spectral features ──────────────────────────────────────────────────
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_fp16):
            linear_spec = compute_linear_spec(waveform)  # (B, 513, T_spec)
            spec_lengths = (wav_lengths.float() / HOP_LENGTH).ceil().long()
            T_spec = linear_spec.shape[-1]
            T_text = input_ids.shape[1]

            # Masks  (convention: posterior_encoder / flow use (B,1,T), text_encoder (B,T,1))
            spec_mask = (torch.arange(T_spec, device=device).unsqueeze(0)
                         < spec_lengths.unsqueeze(1)).unsqueeze(1).float()   # (B,1,T_spec)
            text_mask = (torch.arange(T_text, device=device).unsqueeze(0)
                         < text_lengths.unsqueeze(1)).unsqueeze(2).float()   # (B,T_text,1)

            # ── Text encoder → prior distribution ────────────────────────────
            text_out = self.raw_model.text_encoder(
                input_ids=input_ids,
                padding_mask=text_mask,
            )
            # m_p, logs_p: (B, channels, T_text)
            m_p    = text_out.prior_means.transpose(1, 2)         # (B, C, T_text)
            logs_p = text_out.prior_log_variances.transpose(1, 2) # (B, C, T_text)

            # ── Posterior encoder → latent z ──────────────────────────────────
            z_q, m_q, logs_q = self.raw_model.posterior_encoder(
                linear_spec, spec_mask
            )
            # z_q, m_q, logs_q: (B, C, T_spec)

            # ── Flow (forward): z_q → z_p for KL computation ─────────────────
            z_p = self.raw_model.flow(z_q, spec_mask, reverse=False)  # (B, C, T_spec)

            # ── Monotonic Alignment Search ────────────────────────────────────
            with torch.no_grad():
                # Log-likelihood of each z_p[t] under each prior N(m_p[s], exp(logs_p[s]))
                # shapes: z_p (B,C,T_spec), m_p (B,C,T_text)
                z_exp   = z_p.unsqueeze(3)      # (B, C, T_spec, 1)
                m_exp   = m_p.unsqueeze(2)      # (B, C, 1, T_text)
                lp_exp  = logs_p.unsqueeze(2)   # (B, C, 1, T_text)

                # attn_logprob: (B, T_text, T_spec)
                attn_lp = -0.5 * (
                    (z_exp - m_exp).pow(2) * torch.exp(-2.0 * lp_exp)
                    + 2.0 * lp_exp
                    + math.log(2 * math.pi)
                ).sum(dim=1).transpose(1, 2)   # sum over channels, then (B,T_text,T_spec)

                path = monotonic_alignment_search(
                    attn_lp, text_lengths, spec_lengths
                )  # (B, T_text, T_spec)

            # ── Expand prior to audio length via alignment ─────────────────────
            # (B, C, T_text) @ (B, T_text, T_spec) → (B, C, T_spec)
            m_p_exp    = torch.bmm(m_p,    path)  # (B, C, T_spec)
            logs_p_exp = torch.bmm(logs_p, path)  # (B, C, T_spec)

            # ── KL divergence loss ─────────────────────────────────────────────
            kl = kl_loss(z_p, logs_q, m_p_exp, logs_p_exp, spec_mask)

            # ── Decode z_q → reconstructed waveform (teacher-forced) ──────────
            # Decoder is frozen but gradients flow through it back to flow/enc.
            # Flash Attention enabled (removed the explicit disable — HiFi-GAN
            # decoder has no attention, and the text encoder benefits from FA).
            fake_wav = self.raw_model.decoder(z_q)  # (B, 1, T_wav_gen)
            fake_wav = fake_wav.squeeze(1)           # (B, T_wav_gen)
            # Trim both to the minimum length — generator output length may differ
            # from padded real waveform by a few samples
            t_min    = min(waveform.shape[-1], fake_wav.shape[-1])
            real_wav = waveform[..., :t_min]
            fake_wav = fake_wav[..., :t_min]

            # ── Mel reconstruction loss ────────────────────────────────────────
            m_loss = mel_loss(real_wav, fake_wav, device)

        # ── Discriminator update ───────────────────────────────────────────────
        # Cache r_fmaps here — reused in the generator update for feature matching.
        # Avoids a full discriminator forward on real_wav a second time.
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_fp16):
            r_scores, f_scores, r_fmaps, _ = self.disc(real_wav, fake_wav.detach())
            d_loss = disc_loss(r_scores, f_scores) / self.args.grad_accum

        # NaN guard: if discriminator diverged, reinitialize weights and skip D step
        _disc_nan = not torch.isfinite(d_loss)
        if _disc_nan:
            log.warning("Discriminator loss is NaN — reinitializing disc weights and skipping D step")
            for m in self.disc.modules():
                if isinstance(m, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Linear)):
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None: torch.nn.init.zeros_(m.bias)
        else:
            scaler_d.scale(d_loss).backward()

        # ── Generator update ───────────────────────────────────────────────────
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_fp16):
            _, f_scores_g, _, f_fmaps = self.disc(real_wav.detach(), fake_wav)
            g_adv  = gen_adv_loss(f_scores_g)
            fm     = fmap_loss(r_fmaps, f_fmaps)
            # Fall back to mel+KL only when discriminator is recovering from NaN
            if _disc_nan:
                g_loss = (m_loss * 45.0 + kl * 1.0) / self.args.grad_accum
            else:
                g_loss = (m_loss * 45.0 + kl * 1.0 + g_adv * 1.0 + fm * 1.0) / self.args.grad_accum
        scaler_g.scale(g_loss).backward()

        # Report unscaled per-sample losses (multiply back by grad_accum)
        return {
            "loss/total":   g_loss.item() * self.args.grad_accum,
            "loss/mel":     m_loss.item(),
            "loss/kl":      kl.item(),
            "loss/gen_adv": g_adv.item() * self.args.grad_accum,
            "loss/fmap":    fm.item() * self.args.grad_accum,
            "loss/disc":    d_loss.item() * self.args.grad_accum,
        }

    # ── Main training loop ─────────────────────────────────────────────────────

    def train(self, train_ds, eval_ds, collator):
        from accelerate import Accelerator
        from torch.cuda.amp import GradScaler

        # Pass mixed_precision="no" — we manage fp16 manually via two GradScalers
        # (one per optimizer). Accelerator's built-in GradScaler is single-optimizer
        # and calling optimizer.step() on G triggers unscale_() which then breaks D.
        accelerator = Accelerator(mixed_precision="no")
        device = accelerator.device
        use_fp16 = self.args.fp16 and device.type == "cuda"

        # Two independent scalers — essential for stable GAN fp16 training.
        # If D scaler detects inf/nan it skips only the D step, not G.
        scaler_g = GradScaler(enabled=use_fp16)
        scaler_d = GradScaler(enabled=use_fp16)

        # num_workers=4: parallel data loading keeps A100 fed between batches.
        # persistent_workers avoids re-spawning workers each epoch.
        train_loader = DataLoader(
            train_ds, batch_size=self.args.batch_size, shuffle=True,
            collate_fn=collator, num_workers=4, pin_memory=True,
            drop_last=True, persistent_workers=True,
        )
        eval_loader = DataLoader(
            eval_ds, batch_size=self.args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=2, pin_memory=True,
            drop_last=False, persistent_workers=True,
        )

        # eps=1e-9 matches original VITS paper and all reference implementations
        # (jaywalnut310, coqui-ai, ylacombe). PyTorch default 1e-8 is too large for
        # the small gradient magnitudes typical in VITS flow/encoder layers.
        opt_g = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.args.learning_rate, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.0,
        )
        opt_d = torch.optim.AdamW(
            self.disc.parameters(),
            lr=self.args.learning_rate * 0.25, betas=(0.8, 0.99), eps=1e-9, weight_decay=0.0,
        )

        # ExponentialLR per epoch (gamma=0.999875) — the universal scheduler in every
        # VITS reference (original paper, coqui-ai, jaywalnut310, espnet, ylacombe).
        # CosineAnnealingLR drops LR too aggressively for GAN fine-tuning.
        max_steps = self.args.max_steps or (
            len(train_loader) * self.args.num_epochs // self.args.grad_accum
        )
        sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.999875)
        sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.999875)

        (self.model, self.disc, opt_g, opt_d,
         train_loader, eval_loader) = accelerator.prepare(
            self.model, self.disc, opt_g, opt_d, train_loader, eval_loader
        )
        # DDP wraps self.model — keep a reference to the underlying module for
        # sub-component access (text_encoder, posterior_encoder, flow, decoder).
        self.raw_model = accelerator.unwrap_model(self.model)

        log.info("=" * 60)
        log.info("  Full VITS GAN Training — Turkish language adaptation")
        log.info(f"  Steps: {max_steps}  |  Batch/GPU: {self.args.batch_size}")
        log.info(f"  Grad accum: {self.args.grad_accum}  |  LR: {self.args.learning_rate}")
        log.info(f"  Epochs: {self.args.num_epochs}  |  FP16: {self.args.fp16}")
        log.info("=" * 60)

        # Enable MLflow system metrics (GPU util, CPU, memory) — MLflow ≥ 2.9
        if self.mlflow_run and accelerator.is_main_process:
            try:
                import mlflow
                mlflow.enable_system_metrics_logging()
                self._mlflow_set_tags()
            except Exception as e:
                log.debug(f"MLflow pre-train setup: {e}")

        # Sync all DDP ranks before entering the training loop.
        # Must happen BEFORE any code that is guarded by is_main_process to
        # avoid rank 0 doing extra work (e.g. MLflow) while rank 1 is already
        # waiting on the first all_reduce inside training_step.
        accelerator.wait_for_everyone()

        global_step = 0
        best_mel    = float("inf")
        step_count  = 0
        _loss_history: Dict[str, List[float]] = {}

        # Zero grads before first accumulation window
        opt_g.zero_grad(set_to_none=True)
        opt_d.zero_grad(set_to_none=True)

        for epoch in range(self.args.num_epochs):
            self.model.train()
            self.disc.train()
            epoch_metrics: Dict[str, List[float]] = {}

            for batch in train_loader:
                if self.args.max_steps and global_step >= self.args.max_steps:
                    break

                # ── Coordinated MAS pre-screen ────────────────────────────────────
                # DDP assigns DIFFERENT data to each rank, so rank 0 may hit a bad
                # batch (T_spec < S_text) while rank 1 doesn't. If we let one rank
                # skip and continue, the DDP allreduce inside backward() on the next
                # batch will hang — one rank calls backward() while the other is in
                # a different collective. Fix: agree on the skip BEFORE any compute
                # via a cheap scalar all_reduce, so both ranks skip together.
                wav_lens = batch["wav_lengths"]
                txt_lens = batch["text_lengths"]
                spec_lens = (wav_lens.float() / HOP_LENGTH).ceil().long()
                local_bad = int((spec_lens < txt_lens).any())
                if dist.is_initialized():
                    bad_flag = torch.tensor(local_bad, dtype=torch.long, device=device)
                    dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
                    local_bad = bad_flag.item()
                if local_bad:
                    log.warning(f"Pre-screen skip at step_count {step_count}: "
                                f"min spec_len={spec_lens.min().item()} < "
                                f"max txt_len={txt_lens.max().item()}")
                    continue

                try:
                    metrics = self.training_step(
                        batch, device, scaler_g, scaler_d, opt_g, opt_d, accelerator
                    )
                except (RuntimeError, ValueError) as e:
                    err_str = str(e).lower()
                    if "out of memory" in err_str or "mas impossible" in err_str or \
                            "broadcast" in err_str or "alignment" in err_str:
                        log.warning(f"Skipping batch at step_count {step_count}: {e}")
                        opt_g.zero_grad(set_to_none=True)
                        opt_d.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        gc.collect()
                        # No wait_for_everyone() here — adding a barrier mid-loop
                        # causes a different deadlock when ranks hit different
                        # collectives. OOM is typically symmetric (both ranks OOM)
                        # so just continue independently.
                        continue
                    raise

                step_count += 1
                for k, v in metrics.items():
                    epoch_metrics.setdefault(k, []).append(v)
                    _loss_history.setdefault(k, []).append(v)

                # Optimizer step at accumulation boundary — unscale, clip, step, update scaler
                if step_count % self.args.grad_accum == 0:
                    scaler_g.unscale_(opt_g)
                    scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 1.0
                    )
                    torch.nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
                    scaler_g.step(opt_g)
                    scaler_d.step(opt_d)
                    scaler_g.update()
                    scaler_d.update()
                    opt_g.zero_grad(set_to_none=True)
                    opt_d.zero_grad(set_to_none=True)
                    global_step += 1

                if step_count % self.args.grad_accum != 0:
                    continue  # skip logging/eval until accumulation boundary

                if global_step % self.args.logging_steps == 0 and accelerator.is_main_process:
                    avgs = {k: float(np.mean(vs[-50:])) for k, vs in epoch_metrics.items()}
                    lr   = sched_g.get_last_lr()[0]
                    avgs["train/lr"] = lr
                    log.info(
                        f"[{epoch}] step={global_step}  "
                        + "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in avgs.items()
                                    if k != "train/lr")
                        + f"  lr={lr:.2e}"
                    )
                    if self.mlflow_run:
                        self._mlflow_log(avgs, global_step)

                if global_step % self.args.eval_steps == 0:
                    # All ranks park here while rank 0 does eval + MLflow work.
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        mel_v = self._eval_mel(eval_loader, device)
                        self._last_mel_v = f"{mel_v:.4f}"
                        log.info(f"  Eval mel_loss: {mel_v:.4f}")
                        # On first eval, generate baseline + reference audio and
                        # cache numpy arrays so every subsequent step can include
                        # them in the same folder without re-loading the models.
                        if global_step == self.args.eval_steps:
                            self._cache_comparison_audio(device)
                        self._mlflow_log_step_comparison(device, global_step)
                        self._mlflow_log_loss_plot(_loss_history, global_step)
                        if self.mlflow_run:
                            self._mlflow_log({"eval/mel_loss": mel_v}, global_step)
                        if mel_v < best_mel:
                            best_mel = mel_v
                            self._save(accelerator, "best")
                    accelerator.wait_for_everyone()

                if global_step % self.args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        self._save(accelerator, f"step-{global_step}")
                    accelerator.wait_for_everyone()

            # ExponentialLR steps once per epoch — matches original VITS training protocol
            sched_g.step()
            sched_d.step()

            if self.args.max_steps and global_step >= self.args.max_steps:
                break

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self._save(accelerator, "final")
            self._eval_wer(device)
            self._mlflow_log_audio_artifacts(device, global_step, tag="final")
            self._mlflow_log_loss_plot(_loss_history, global_step)
            self._mlflow_log_model(accelerator, "final")
        accelerator.wait_for_everyone()
        log.info(f"Training complete. Best eval mel: {best_mel:.4f}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _eval_mel(self, loader, device) -> float:
        self.model.eval(); self.disc.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                try:
                    wav = batch["waveform"].to(device)
                    wav_lens = batch["wav_lengths"].to(device)
                    spec = compute_linear_spec(wav)
                    spec_lengths = (wav_lens.float() / HOP_LENGTH).ceil().long()
                    T_spec = spec.shape[-1]
                    mask = (torch.arange(T_spec, device=device).unsqueeze(0)
                            < spec_lengths.unsqueeze(1)).unsqueeze(1).float()
                    z_q, _, _ = self.raw_model.posterior_encoder(spec, mask)
                    fake = self.raw_model.decoder(z_q).squeeze(1)
                    losses.append(mel_loss(wav, fake, device).item())
                except Exception as e:
                    log.debug(f"Eval batch skipped: {e}")
        self.model.train(); self.disc.train()
        return float(np.mean(losses)) if losses else 999.0

    def _eval_wer(self, device) -> None:
        import soundfile as sf
        out_dir = Path(self.args.output_dir) / "eval-audio"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        paths = []
        with torch.no_grad():
            for i, text in enumerate(EVAL_TEXTS):
                try:
                    inputs = self.tokenizer(text, return_tensors="pt").to(device)
                    torch.manual_seed(555)
                    out = self.model(**inputs)
                    wav = out.waveform.squeeze().cpu().numpy()
                    p = out_dir / f"eval_{i:02d}.wav"
                    sf.write(str(p), wav, samplerate=TARGET_SR)
                    paths.append((text, str(p)))
                except Exception as e:
                    log.warning(f"TTS eval failed for sample {i}: {e}")

        if not paths:
            return
        try:
            import whisper
            from jiwer import cer as j_cer, wer as j_wer
            asr = whisper.load_model("small", device="cpu")
            wers, cers = [], []
            for ref, p in paths:
                hyp = asr.transcribe(p, language="tr")["text"].strip().lower()
                wers.append(j_wer(ref, hyp))
                cers.append(j_cer(ref, hyp))
                log.info(f"  REF: {ref[:60]}  HYP: {hyp[:60]}")
            log.info(f"  WER={np.mean(wers):.2%}  CER={np.mean(cers):.2%}")
            if self.mlflow_run:
                self._mlflow_log({
                    "eval/final_wer": float(np.mean(wers)),
                    "eval/final_cer": float(np.mean(cers)),
                }, step=0)
        except Exception as e:
            log.warning(f"WER eval skipped: {e}")

    def _save(self, accelerator, tag: str) -> None:
        out = Path(self.args.output_dir) / tag
        out.mkdir(parents=True, exist_ok=True)
        unwrapped_g = accelerator.unwrap_model(self.model)
        unwrapped_d = accelerator.unwrap_model(self.disc)
        unwrapped_g.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        torch.save(unwrapped_d.state_dict(), str(out / "discriminator.pt"))
        log.info(f"  Checkpoint → {out}")

        if not self.mlflow_run:
            return
        try:
            import mlflow
            for fname in ("config.json", "tokenizer_config.json",
                          "tokenizer.json", "special_tokens_map.json"):
                p = out / fname
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path=f"checkpoints/{tag}")
            run_id = mlflow.active_run().info.run_id
            mv = mlflow.register_model(
                f"runs:/{run_id}/checkpoints/{tag}",
                "mms-tts-turkish",
            )
            client = mlflow.MlflowClient()
            client.set_model_version_tag("mms-tts-turkish", mv.version, "checkpoint_tag", tag)
            client.set_model_version_tag("mms-tts-turkish", mv.version,
                                         "checkpoint_pvc_path", str(out))
            if tag == "final":
                client.set_model_version_tag("mms-tts-turkish", mv.version, "stage", "final")
            log.info(f"  Registered mms-tts-turkish v{mv.version} ({tag})")
        except Exception as e:
            log.warning(f"Checkpoint registration failed ({tag}): {e}")

    def _mlflow_log(self, metrics: Dict, step: int) -> None:
        try:
            import mlflow
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            pass

    def _mlflow_log_audio_artifacts(
        self, device: torch.device, step: int, tag: str = "train"
    ) -> None:
        """Generate audio for fixed test sentences and log to MLflow as artifacts + spectrograms."""
        if not self.mlflow_run:
            return
        try:
            import mlflow
            import soundfile as sf
            import tempfile, os

            self.model.eval()
            audio_dir = Path(tempfile.mkdtemp())

            wavs_logged = []
            with torch.no_grad():
                for i, text in enumerate(EVAL_TEXTS[:5]):
                    try:
                        inputs = self.tokenizer(text, return_tensors="pt").to(device)
                        torch.manual_seed(555)
                        out = self.model(**inputs)
                        wav = out.waveform.squeeze().cpu().numpy()
                        path = audio_dir / f"step{step:06d}_{tag}_{i:02d}.wav"
                        sf.write(str(path), wav, samplerate=TARGET_SR)
                        wavs_logged.append((text, str(path), wav))
                    except Exception as e:
                        log.debug(f"Audio gen failed sample {i}: {e}")

            # Log WAVs
            for _, path, _ in wavs_logged:
                mlflow.log_artifact(path, artifact_path=f"audio/{tag}/step{step:06d}")

            # Log mel spectrogram images (shows quality progression visually)
            self._mlflow_log_spectrograms(wavs_logged, step, tag, audio_dir)

            self.model.train()
        except Exception as e:
            log.debug(f"MLflow audio artifact logging failed: {e}")
            self.model.train()

    def _cache_comparison_audio(self, device: torch.device) -> None:
        """Generate baseline (mms-tts-eng) and reference (mms-tts-tur) audio once
        and cache as numpy arrays so every eval step can include them without
        reloading the models."""
        try:
            import soundfile as sf
            from transformers import VitsModel, VitsTokenizer

            self._comparison_wavs: list = []  # [(text, baseline_np, reference_np)]

            log.info("Caching baseline (mms-tts-eng) audio …")
            base_model = VitsModel.from_pretrained(
                "facebook/mms-tts-eng", cache_dir=self.args.hf_cache
            ).to(device).eval()
            base_tok = VitsTokenizer.from_pretrained(
                "facebook/mms-tts-eng", cache_dir=self.args.hf_cache
            )

            log.info("Caching reference (mms-tts-tur) audio …")
            ref_model = VitsModel.from_pretrained(
                "facebook/mms-tts-tur", cache_dir=self.args.hf_cache
            ).to(device).eval()
            ref_tok = VitsTokenizer.from_pretrained(
                "facebook/mms-tts-tur", cache_dir=self.args.hf_cache
            )

            with torch.no_grad():
                for text in EVAL_TEXTS[:5]:
                    try:
                        b_wav = base_model(
                            **base_tok(text, return_tensors="pt").to(device)
                        ).waveform.squeeze().cpu().numpy()
                    except Exception:
                        b_wav = np.zeros(TARGET_SR, dtype=np.float32)
                    try:
                        r_wav = ref_model(
                            **ref_tok(text, return_tensors="pt").to(device)
                        ).waveform.squeeze().cpu().numpy()
                    except Exception:
                        r_wav = np.zeros(TARGET_SR, dtype=np.float32)
                    self._comparison_wavs.append((text, b_wav, r_wav))

            del base_model, ref_model
            torch.cuda.empty_cache()
            log.info(f"Cached {len(self._comparison_wavs)} comparison pairs.")
        except Exception as e:
            log.warning(f"Comparison audio cache failed: {e}")
            self._comparison_wavs = []

    def _get_whisper(self):
        """Lazy-load Whisper-small on CPU — shared across eval steps."""
        if not hasattr(self, "_whisper_mdl"):
            try:
                import whisper
                self._whisper_mdl = whisper.load_model("small", device="cpu")
                log.info("Whisper-small loaded on CPU for per-step WER/CER")
            except Exception as e:
                log.warning(f"Whisper unavailable: {e}")
                self._whisper_mdl = None
        return self._whisper_mdl

    def _mlflow_log_step_comparison(self, device: torch.device, step: int) -> None:
        """Log all three audio tracks (baseline / fine-tuned / reference) into a
        single MLflow folder per step so reviewers can do A/B/C comparison.

        Folder layout per step:
            audio/step{N:06d}/
                text00_baseline.wav    ← mms-tts-eng (English model, unchanged)
                text00_finetuned.wav   ← our model at step N
                text00_reference.wav   ← mms-tts-tur (gold-standard Turkish)
                spectrogram.png        ← mel spectrograms of all three
        """
        if not self.mlflow_run:
            return
        try:
            import mlflow, soundfile as sf, tempfile
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import librosa, librosa.display

            _has_trace = hasattr(mlflow, "trace")
            self.model.eval()
            artifact_path = f"audio/step{step:06d}"
            tmp = Path(tempfile.mkdtemp())

            comparison = getattr(self, "_comparison_wavs", [])
            n = min(len(EVAL_TEXTS), 5)
            fig_rows = n
            fig, axes = plt.subplots(fig_rows, 3, figsize=(18, 3 * fig_rows))
            if fig_rows == 1:
                axes = [axes]
            col_titles = ["Baseline (mms-tts-eng)", f"Fine-tuned (step {step})", "Reference (mms-tts-tur)"]

            step_wers, step_cers = [], []
            whisper_mdl = self._get_whisper()

            with torch.no_grad():
                for i, text in enumerate(EVAL_TEXTS[:n]):
                    # Fine-tuned audio — traced
                    try:
                        t0 = time.perf_counter()
                        inputs = self.tokenizer(text, return_tensors="pt").to(device)
                        torch.manual_seed(555)

                        if _has_trace:
                            with mlflow.start_span(name="vits_tts_pipeline", span_type="CHAIN") as root:
                                root.set_inputs({"text": text, "training_step": step})
                                with mlflow.start_span(name="tokenise", span_type="PARSER") as sp:
                                    sp.set_inputs({"text": text})
                                    sp.set_outputs({"n_tokens": inputs["input_ids"].shape[-1]})
                                with mlflow.start_span(name="model_forward", span_type="LLM") as sp:
                                    sp.set_inputs({"input_tokens": inputs["input_ids"].shape[-1]})
                                    out = self.model(**inputs)
                                    ft_wav = out.waveform.squeeze().cpu().numpy()
                                    elapsed = time.perf_counter() - t0
                                    duration = len(ft_wav) / TARGET_SR
                                    rtf = elapsed / max(duration, 1e-6)
                                    sp.set_outputs({"duration_s": round(duration, 3),
                                                    "sample_rate": TARGET_SR,
                                                    "rtf": round(rtf, 4)})
                                root.set_outputs({"duration_s": round(duration, 3),
                                                  "rtf": round(rtf, 4),
                                                  "elapsed_s": round(elapsed, 3)})
                        else:
                            out = self.model(**inputs)
                            ft_wav = out.waveform.squeeze().cpu().numpy()
                            elapsed = time.perf_counter() - t0
                            duration = len(ft_wav) / TARGET_SR
                            rtf = elapsed / max(duration, 1e-6)

                        mlflow.log_metrics({
                            f"rtf/text{i:02d}": rtf,
                            f"duration_s/text{i:02d}": duration,
                        }, step=step)

                        # WER/CER via Whisper
                        if whisper_mdl is not None:
                            try:
                                from jiwer import wer as j_wer, cer as j_cer
                                hyp = whisper_mdl.transcribe(
                                    ft_wav.astype(np.float32),
                                    language="tr",
                                    initial_prompt="Türkçe konuşma.",
                                )["text"].strip().lower()
                                ref = text.strip().lower()
                                wer_v = j_wer(ref, hyp)
                                cer_v = j_cer(ref, hyp)
                                step_wers.append(wer_v)
                                step_cers.append(cer_v)
                                mlflow.log_metrics({
                                    f"wer/text{i:02d}": wer_v,
                                    f"cer/text{i:02d}": cer_v,
                                }, step=step)
                                log.info(f"  [text{i:02d}] WER={wer_v:.3f} CER={cer_v:.3f} hyp: {hyp[:60]}")
                            except Exception as e:
                                log.debug(f"WER/CER failed text{i}: {e}")

                    except Exception:
                        ft_wav = np.zeros(TARGET_SR, dtype=np.float32)

                    b_wav = comparison[i][1] if i < len(comparison) else np.zeros(TARGET_SR, dtype=np.float32)
                    r_wav = comparison[i][2] if i < len(comparison) else np.zeros(TARGET_SR, dtype=np.float32)

                    for fname, wav in [
                        (f"text{i:02d}_baseline.wav",  b_wav),
                        (f"text{i:02d}_finetuned.wav", ft_wav),
                        (f"text{i:02d}_reference.wav", r_wav),
                    ]:
                        sf.write(str(tmp / fname), wav, samplerate=TARGET_SR)
                        mlflow.log_artifact(str(tmp / fname), artifact_path=artifact_path)

                    # Spectrogram row
                    for col, (wav, title) in enumerate(zip([b_wav, ft_wav, r_wav], col_titles)):
                        ax = axes[i][col]
                        mel = librosa.feature.melspectrogram(
                            y=wav, sr=TARGET_SR, n_fft=N_FFT,
                            hop_length=HOP_LENGTH, n_mels=N_MELS,
                        )
                        librosa.display.specshow(
                            librosa.power_to_db(mel, ref=np.max),
                            ax=ax, sr=TARGET_SR, hop_length=HOP_LENGTH,
                        )
                        if i == 0:
                            ax.set_title(title, fontsize=9)
                        ax.set_ylabel(f"s{i}", fontsize=7)

            # Aggregate WER/CER means
            if step_wers:
                mlflow.log_metrics({
                    "wer/mean": float(np.mean(step_wers)),
                    "cer/mean": float(np.mean(step_cers)),
                }, step=step)

            fig.suptitle(
                f"Step {step} — mel_loss={getattr(self, '_last_mel_v', '?')}"
                + (f"  WER={np.mean(step_wers):.3f}  CER={np.mean(step_cers):.3f}"
                   if step_wers else ""),
                fontsize=10,
            )
            plt.tight_layout()
            spec_path = tmp / f"spectrogram_step{step:06d}.png"
            fig.savefig(str(spec_path), dpi=80, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(spec_path), artifact_path=artifact_path)

            self.model.train()
        except Exception as e:
            log.debug(f"Step comparison logging failed: {e}")
            self.model.train()

    def _mlflow_log_spectrograms(
        self, wavs: list, step: int, tag: str, out_dir: Path
    ) -> None:
        """Save mel spectrogram PNG grid and log to MLflow."""
        try:
            import mlflow
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import librosa, librosa.display

            n = len(wavs)
            if n == 0:
                return
            fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
            if n == 1:
                axes = [axes]

            for ax, (text, _, wav) in zip(axes, wavs):
                mel = librosa.feature.melspectrogram(
                    y=wav, sr=TARGET_SR, n_fft=N_FFT,
                    hop_length=HOP_LENGTH, n_mels=N_MELS
                )
                librosa.display.specshow(
                    librosa.power_to_db(mel, ref=np.max),
                    ax=ax, sr=TARGET_SR, hop_length=HOP_LENGTH,
                    x_axis="time", y_axis="mel",
                )
                ax.set_title(text[:60], fontsize=8)
                ax.set_ylabel("")

            fig.suptitle(f"Mel spectrograms — step {step}", fontsize=10, y=1.01)
            plt.tight_layout()
            img_path = out_dir / f"spectrograms_step{step:06d}.png"
            fig.savefig(str(img_path), dpi=80, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(img_path), artifact_path=f"spectrograms/{tag}")
        except Exception as e:
            log.debug(f"Spectrogram logging failed: {e}")

    def _mlflow_log_loss_plot(self, history: Dict[str, List[float]], step: int) -> None:
        """Log a loss curve PNG to MLflow."""
        if not self.mlflow_run:
            return
        try:
            import mlflow
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import tempfile

            keys = ["loss/mel", "loss/kl", "loss/gen_adv", "loss/disc"]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for ax, key in zip(axes.flat, keys):
                vals = history.get(key, [])
                if vals:
                    # Smooth with running average
                    window = max(1, len(vals) // 50)
                    smooth = np.convolve(vals, np.ones(window) / window, mode="valid")
                    ax.plot(vals, alpha=0.3, color="steelblue", linewidth=0.5)
                    ax.plot(smooth, color="steelblue", linewidth=1.5)
                ax.set_title(key.split("/")[-1], fontsize=10)
                ax.set_xlabel("step"); ax.grid(alpha=0.3)
            fig.suptitle(f"Training losses — step {step}", fontsize=12)
            plt.tight_layout()
            tmp = Path(tempfile.mkdtemp()) / f"loss_curves_step{step:06d}.png"
            fig.savefig(str(tmp), dpi=80, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(tmp), artifact_path="plots")
        except Exception as e:
            log.debug(f"Loss plot logging failed: {e}")

    def _mlflow_log_model(self, accelerator, tag: str = "final") -> None:
        """Register the trained model to MLflow Model Registry with signature."""
        if not self.mlflow_run:
            return
        try:
            import mlflow
            import mlflow.transformers

            unwrapped = accelerator.unwrap_model(self.model)
            pipe = {"model": unwrapped, "tokenizer": self.tokenizer}
            sig_input  = mlflow.models.infer_signature(
                {"text": ["merhaba dünya"]},
                {"waveform": np.zeros((1, TARGET_SR), dtype=np.float32)},
            )
            mlflow.transformers.log_model(
                transformers_model=pipe,
                artifact_path="model",
                task="text-to-speech",
                signature=sig_input,
                registered_model_name="mms-tts-turkish",
            )
            log.info("Model registered in MLflow Model Registry: mms-tts-turkish")
        except Exception as e:
            log.warning(f"Model registry logging failed (non-critical): {e}")

    def _mlflow_log_reference_audio(self, device: torch.device) -> None:
        """Generate audio with facebook/mms-tts-tur (gold-standard Turkish TTS) and log to MLflow.

        Logged once per run under audio/reference/ so reviewers can compare:
            audio/baseline/  — mms-tts-eng on Turkish text (mangled, wrong accent)
            audio/reference/ — mms-tts-tur  (target quality)
            audio/eval/      — our fine-tuned model at each eval step
        """
        if not self.mlflow_run:
            return
        try:
            import mlflow
            import soundfile as sf
            import tempfile
            from transformers import VitsModel, VitsTokenizer

            log.info("Logging reference audio from facebook/mms-tts-tur …")
            ref_model = VitsModel.from_pretrained(
                "facebook/mms-tts-tur",
                cache_dir=self.args.hf_cache,
            ).to(device)
            ref_tok = VitsTokenizer.from_pretrained(
                "facebook/mms-tts-tur",
                cache_dir=self.args.hf_cache,
            )
            ref_model.eval()

            audio_dir = Path(tempfile.mkdtemp())
            wavs_logged = []
            with torch.no_grad():
                for i, text in enumerate(EVAL_TEXTS[:5]):
                    try:
                        inputs = ref_tok(text, return_tensors="pt").to(device)
                        torch.manual_seed(555)
                        out = ref_model(**inputs)
                        wav = out.waveform.squeeze().cpu().numpy()
                        sr = ref_model.config.sampling_rate  # 16000
                        path = audio_dir / f"reference_{i:02d}.wav"
                        sf.write(str(path), wav, samplerate=sr)
                        mlflow.log_artifact(str(path), artifact_path="audio/reference")
                        wavs_logged.append((text, str(path), wav))
                        log.info(f"  Reference [{i}] logged: {text[:50]}")
                    except Exception as e:
                        log.warning(f"Reference audio gen failed for sample {i}: {e}")

            # Spectrogram grid for the reference set
            if wavs_logged:
                self._mlflow_log_spectrograms(wavs_logged, step=0, tag="reference", out_dir=audio_dir)

            # Free VRAM — ref model is only needed once
            del ref_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.info("Reference audio logged to MLflow.")
        except Exception as e:
            log.warning(f"Reference audio logging failed (non-critical): {e}")

    def _mlflow_set_tags(self) -> None:
        """Log run metadata: hardware, dataset, strategy."""
        if not self.mlflow_run:
            return
        try:
            import mlflow
            world = int(os.getenv("WORLD_SIZE", 1))
            tags = {
                "base_model":        self.args.base_model,
                "dataset":           "afkfatih/turkish-tts-combined-raw",
                "strategy":          "full_vits_gan",
                "frozen":            "decoder/HiFiGAN",
                "trained":           "text_encoder,posterior_encoder,flow",
                "losses":            "mel×45+KL×1.0+gen_adv+fmap",
                "effective_batch":   str(self.args.batch_size * self.args.grad_accum * world),
                "n_gpus":            str(world),
                "platform":          "openshift-ai/kubeflow-trainer-v2",
            }
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_properties(0)
                tags["gpu"]  = gpu.name
                tags["vram"] = f"{gpu.total_memory // 1024**3}GB"
            mlflow.set_tags(tags)
        except Exception as e:
            log.debug(f"MLflow tag setting failed: {e}")

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def parse_args():
    p = argparse.ArgumentParser(
        description="VITS GAN fine-tuning — all args fall back to env vars"
    )
    # Storage
    p.add_argument("--dataset_dir",   type=Path,  default=Path(_env("DATASET_DIR",    "/data/tts/mms-dataset")))
    p.add_argument("--output_dir",    type=Path,  default=Path(_env("CHECKPOINT_DIR", "/data/tts/checkpoints/mms-turkish")))
    p.add_argument("--base_model",                default=_env("BASE_MODEL",          "facebook/mms-tts-eng"))
    p.add_argument("--hf_cache",                  default=_env("HF_HOME",             "/data/tts/hf-cache"))
    # Hyperparameters
    p.add_argument("--num_epochs",    type=int,   default=int(_env("NUM_EPOCHS",    "50")))
    p.add_argument("--batch_size",    type=int,   default=int(_env("BATCH_SIZE",    "8")))
    p.add_argument("--grad_accum",    type=int,   default=int(_env("GRAD_ACCUM",    "4")))
    p.add_argument("--learning_rate", type=float, default=float(_env("LEARNING_RATE", "2e-4")))
    p.add_argument("--max_steps",     type=int,   default=int(_env("MAX_STEPS")) if _env("MAX_STEPS") else None,
                   help="Override num_epochs with a fixed step count")
    p.add_argument("--max_train_samples", type=int, default=int(_env("MAX_TRAIN_SAMPLES")) if _env("MAX_TRAIN_SAMPLES") else None,
                   help="Subset training set for faster iterations")
    p.add_argument("--eval_steps",    type=int,   default=int(_env("EVAL_STEPS",    "500")))
    p.add_argument("--save_steps",    type=int,   default=int(_env("SAVE_STEPS",    "1000")))
    p.add_argument("--logging_steps", type=int,   default=int(_env("LOGGING_STEPS", "50")))
    p.add_argument("--fp16",          action="store_true", default=True)
    p.add_argument("--max_wav_seconds", type=float, default=8.0)
    # Checkpointing
    p.add_argument("--resume_from",      type=Path, default=Path(_env("RESUME_FROM")) if _env("RESUME_FROM") else None)
    p.add_argument("--disc_checkpoint",  type=Path, default=Path(_env("DISC_CHECKPOINT")) if _env("DISC_CHECKPOINT") else None)
    # MLflow
    p.add_argument("--mlflow_uri",        default=_env("MLFLOW_TRACKING_URI"))
    p.add_argument("--mlflow_experiment", default=_env("MLFLOW_EXPERIMENT", "mms-turkish-tts"))
    p.add_argument("--run_name",          default=None)
    # Post-training evaluation (rank 0 only)
    p.add_argument("--run_eval", action="store_true",
                   default=_env("RUN_EVAL", "false").lower() == "true",
                   help="Run evaluate.py after training completes (rank 0 only)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_cache

    from datasets import load_from_disk
    from transformers import VitsModel, VitsTokenizer
    import torch.nn as nn

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_run = None
    if args.mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            run_name = args.run_name or f"vits-gan-tr-b{args.batch_size}x{args.grad_accum}"
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "base_model":    args.base_model,
                "strategy":      "full_vits_gan",
                "dataset":       "afkfatih/turkish-tts-combined-raw",
                "batch_size":    args.batch_size,
                "grad_accum":    args.grad_accum,
                "learning_rate": args.learning_rate,
                "num_epochs":    args.num_epochs,
                "max_steps":     args.max_steps,
            })
            log.info(f"MLflow: {args.mlflow_uri}  run={run_name}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # ── Tokenizer (expanded for Turkish) ──────────────────────────────────────
    tok_path = args.dataset_dir / "tokenizer"
    if tok_path.exists():
        tokenizer = VitsTokenizer.from_pretrained(str(tok_path))
        log.info(f"Loaded expanded tokenizer: {len(tokenizer)} tokens")
    else:
        tokenizer = VitsTokenizer.from_pretrained(args.base_model)
        log.warning("Expanded tokenizer not found — using base English tokenizer")

    # ── Generator model ───────────────────────────────────────────────────────
    src = str(args.resume_from) if args.resume_from else args.base_model
    log.info(f"Loading generator: {src}")
    model = VitsModel.from_pretrained(src)

    if len(tokenizer) != model.config.vocab_size:
        old, new = model.config.vocab_size, len(tokenizer)
        log.info(f"Resizing embed_tokens: {old} → {new}")
        old_emb = model.text_encoder.embed_tokens
        new_emb = nn.Embedding(new, old_emb.embedding_dim, padding_idx=old_emb.padding_idx)
        with torch.no_grad():
            new_emb.weight[:old] = old_emb.weight
            nn.init.normal_(new_emb.weight[old:], mean=0.0, std=0.02)
        model.text_encoder.embed_tokens = new_emb
        model.config.vocab_size = new

    log.info(f"Generator params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Discriminator ─────────────────────────────────────────────────────────
    discriminator = VitsDiscriminator()
    if args.disc_checkpoint and args.disc_checkpoint.exists():
        log.info(f"Loading discriminator from {args.disc_checkpoint}")
        discriminator.load_state_dict(torch.load(str(args.disc_checkpoint), map_location="cpu"))
    else:
        log.info("Discriminator: random init (standard VITS approach)")
    log.info(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    log.info("Loading preprocessed dataset...")
    train_ds = load_from_disk(str(args.dataset_dir / "train"))
    val_ds   = load_from_disk(str(args.dataset_dir / "validation"))
    if args.max_train_samples and args.max_train_samples < len(train_ds):
        train_ds = train_ds.select(range(args.max_train_samples))
        log.info(f"Subsetted training set to {args.max_train_samples} samples")
    log.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    if "waveform" not in train_ds.column_names:
        log.error(
            "Dataset missing 'waveform' column — re-run preprocess_mms.py with "
            "--save_waveforms flag (FORCE_PREPROCESS=true in entrypoint)."
        )
        sys.exit(1)

    collator = VitsGANCollator(
        pad_id=tokenizer.pad_token_id or 0,
        max_wav_len=int(TARGET_SR * args.max_wav_seconds),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = VitsGANTrainer(
        model=model,
        tokenizer=tokenizer,
        discriminator=discriminator,
        args=args,
        mlflow_run=mlflow_run,
    )
    trainer.train(train_ds, val_ds, collator)

    if mlflow_run:
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass

    # ── Post-training evaluation (rank 0 only) ────────────────────────────────
    if args.run_eval and int(os.environ.get("RANK", "0")) == 0:
        import subprocess
        ckpt_dir = args.output_dir
        for subdir in ("best", "final"):
            candidate = ckpt_dir / subdir
            if candidate.exists():
                ckpt_dir = candidate
                break

        log.info(f"Running evaluation on {ckpt_dir}")
        eval_cmd = [sys.executable, "/scripts/evaluate.py", "--checkpoint", str(ckpt_dir),
                    "--out_dir", str(args.output_dir.parent / "eval-outputs"),
                    "--hf_cache", args.hf_cache]
        if args.mlflow_uri:
            eval_cmd += ["--mlflow_uri", args.mlflow_uri,
                         "--mlflow_experiment", args.mlflow_experiment]
        result = subprocess.run(eval_cmd, check=False)
        if result.returncode != 0:
            log.warning(f"Evaluation exited with code {result.returncode} — training artifacts are still saved")


if __name__ == "__main__":
    main()
