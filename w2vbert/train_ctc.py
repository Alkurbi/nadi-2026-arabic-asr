"""Fine-tune facebook/w2v-bert-2.0 with a CTC head on NADI dialectal Arabic.

Runs in the Windows .venv. Targets = officially-normalized transcripts.
Uses DYNAMIC duration-based batching: each batch is packed to a fixed total-audio
budget (seconds), so peak Conformer O(T^2) activation memory is bounded regardless of
clip length, no padding is wasted, and no data is dropped. Saves to w2vbert/best/.
"""
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from norm import normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "finetune", "data")
MODEL = os.environ.get("INIT_MODEL", "facebook/w2v-bert-2.0")  # can be a local checkpoint (stage 2)
SR = 16000


def ctc_ids_from_trainer_logits(logits, pad_token_id):
    """Argmax Trainer logits without decoding its -100 concat padding as token 0."""
    logits = np.asarray(logits)
    padded_frames = np.all(logits == -100, axis=-1)
    ids = np.argmax(logits, axis=-1)
    ids[padded_frames] = pad_token_id
    return ids


def host_path(p):
    return p.replace("/data", DATA.replace("\\", "/"), 1) if p.startswith("/data") else p


def load_manifest(names, dialect=""):
    rows = []
    for name in names:
        for l in open(os.path.join(DATA, name), encoding="utf-8"):
            r = json.loads(l)
            if dialect and f"/train/{dialect}/" not in r["audio_filepath"].replace("\\", "/"):
                continue
            rows.append((host_path(r["audio_filepath"]), normalize(r["text"]), float(r["duration"])))
    return rows


def split_holdout(rows, size, seed=0):
    """Return deterministic train/dev partitions without changing manifest order."""
    if not 0 <= size < len(rows):
        raise ValueError(f"DEV_HOLDOUT must be in [0, {len(rows) - 1}], got {size}")
    if not size:
        return rows, []
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    dev_indices = set(indices[:size])
    return ([row for i, row in enumerate(rows) if i not in dev_indices],
            [row for i, row in enumerate(rows) if i in dev_indices])


class ASRDataset(Dataset):
    def __init__(self, rows, processor):
        self.rows = [r for r in rows if r[1]]
        self.p = processor
        self.durations = [r[2] for r in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, text, _ = self.rows[i]
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(1)
        feat = self.p.feature_extractor(wav, sampling_rate=SR).input_features[0]
        labels = self.p.tokenizer(text).input_ids
        return {"input_features": feat, "labels": labels}


class DurationBatchSampler(Sampler):
    """Pack indices into batches whose total audio duration <= budget seconds."""
    def __init__(self, durations, budget=45.0, shuffle=True, seed=0):
        self.durations = durations
        self.budget = budget
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self._batches = self._build()

    def _build(self):
        order = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        batches, cur, cur_max = [], [], 0.0
        for i in order:
            d = self.durations[i]
            # total padded cost = (len(cur)+1) * max_dur_in_batch
            new_max = max(cur_max, d)
            if cur and (len(cur) + 1) * new_max > self.budget:
                batches.append(cur)
                cur, cur_max = [i], d
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        batches = list(self._batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self):
        return len(self._batches)


@dataclass
class Collator:
    processor: object

    def __call__(self, batch):
        feats = [{"input_features": b["input_features"]} for b in batch]
        labels = [{"input_ids": b["labels"]} for b in batch]
        out = self.processor.feature_extractor.pad(feats, return_tensors="pt")
        lab = self.processor.tokenizer.pad(labels, return_tensors="pt")
        out["labels"] = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        return out


def main():
    from transformers import (SeamlessM4TFeatureExtractor, Trainer,
                              TrainingArguments, Wav2Vec2BertForCTC,
                              Wav2Vec2CTCTokenizer, Wav2Vec2BertProcessor)
    from torch.utils.data import DataLoader
    import evaluate

    tok = Wav2Vec2CTCTokenizer(os.path.join(HERE, "vocab.json"), unk_token="[UNK]",
                               pad_token="[PAD]", word_delimiter_token="|")
    fe = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    processor = Wav2Vec2BertProcessor(feature_extractor=fe, tokenizer=tok)
    processor.save_pretrained(os.path.join(HERE, "processor"))

    train_names = os.environ.get("TRAIN_MANIFESTS", "train_manifest.json").split(",")
    dialect = os.environ.get("DIALECT", "")
    train_rows = load_manifest(train_names, dialect=dialect)
    holdout = int(os.environ.get("DEV_HOLDOUT", 0))
    if holdout:
        train_rows, dev_rows = split_holdout(train_rows, holdout, int(os.environ.get("SEED", 0)))
    else:
        dev_rows = load_manifest([os.environ.get("DEV_MANIFEST", "dev_manifest.json")], dialect=dialect)
    train_ds = ASRDataset(train_rows, processor)
    dev_ds = ASRDataset(dev_rows, processor)
    print(f"dialect={dialect or 'ALL'} train={len(train_ds)} dev={len(dev_ds)}", flush=True)

    model = Wav2Vec2BertForCTC.from_pretrained(
        MODEL, vocab_size=len(tok), ctc_loss_reduction="mean",
        ctc_zero_infinity=True, pad_token_id=tok.pad_token_id, add_adapter=True,
    )
    train_mode = os.environ.get("TRAIN_MODE", "full")
    if train_mode == "adapter_head":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_("adapter" in name or name.startswith("lm_head"))
    elif train_mode != "full":
        raise ValueError(f"unknown TRAIN_MODE={train_mode!r}; expected 'full' or 'adapter_head'")
    if train_mode == "full":
        model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"train_mode={train_mode} trainable={trainable / 1e6:.1f}M/{total / 1e6:.1f}M", flush=True)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        ids = ctc_ids_from_trainer_logits(pred.predictions, tok.pad_token_id)
        pred_str = processor.batch_decode(ids)
        labels = pred.label_ids
        labels[labels == -100] = tok.pad_token_id
        ref_str = processor.batch_decode(labels, group_tokens=False)
        return {"wer": wer_metric.compute(predictions=[p or "@" for p in pred_str], references=ref_str)}

    budget = float(os.environ.get("BUDGET", 45))
    collator = Collator(processor)
    train_sampler = DurationBatchSampler(train_ds.durations, budget=budget, shuffle=True)
    print(f"dynamic batches/epoch: {len(train_sampler)} (budget {budget}s)", flush=True)

    args = TrainingArguments(
        output_dir=os.environ.get("OUTDIR", os.path.join(HERE, "ckpt")),
        per_device_eval_batch_size=int(os.environ.get("EBS", 4)),
        eval_strategy="steps", eval_steps=int(os.environ.get("EVAL_STEPS", 500)),
        save_steps=int(os.environ.get("EVAL_STEPS", 500)), save_total_limit=2,
        logging_steps=25, num_train_epochs=float(os.environ.get("EPOCHS", 10)),
        learning_rate=float(os.environ.get("LR", 1e-4)),
        warmup_steps=int(os.environ.get("WARMUP", 300)),
        bf16=True, gradient_checkpointing=train_mode == "full",
        dataloader_num_workers=int(os.environ.get("NW", 2)),
        metric_for_best_model="wer", greater_is_better=False, load_best_model_at_end=True,
        remove_unused_columns=False, report_to=[],
    )

    class DynTrainer(Trainer):
        def get_train_dataloader(self):
            return DataLoader(self.train_dataset, batch_sampler=train_sampler,
                              collate_fn=collator, num_workers=args.dataloader_num_workers,
                              pin_memory=True)

    trainer = DynTrainer(model=model, args=args, data_collator=collator,
                         train_dataset=train_ds, eval_dataset=dev_ds,
                         compute_metrics=compute_metrics,
                         processing_class=processor.feature_extractor)
    trainer.train(resume_from_checkpoint=bool(os.environ.get("RESUME")))
    savedir = os.path.join(HERE, os.environ.get("SAVEDIR", "best"))  # never clobber best/ unless asked
    trainer.save_model(savedir)
    processor.save_pretrained(savedir)
    print(f"SAVED {savedir}", flush=True)


if __name__ == "__main__":
    main()
