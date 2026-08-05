# Compute-constrained multidialectal Arabic ASR

Reproducibility code for NADI 2026 Subtask 1.1 (Robust Arabic ASR). The reported
system fine-tunes `facebook/w2v-bert-2.0` with a CTC head, decodes with a 5-gram
KenLM language model, and combines diverse recognizers with confusion-network
ROVER or minimum Bayes risk (MBR) selection.

The repository intentionally contains **no competition audio, transcripts,
manifests, model weights, checkpoints, or predictions**. Obtain the NADI data
under its own terms and keep it outside version control.

## Reported result

On our fixed 800-utterance holdout (100 utterances per dialect), the strongest
single recognizer reached 46.12% macro WER and the selected four-system ensemble
reached 42.59% macro WER. See [the system description](docs/nadi-system-paper.md)
for context.

## Environment

Python 3.11 and a CUDA-capable GPU were used. Create an environment and install
PyTorch for your CUDA version first, followed by the remaining dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

KenLM decoding also requires a working KenLM installation compatible with
`pyctcdecode`.

## Expected private data layout

Create JSON Lines manifests under `finetune/data/`. Each row must contain an
audio path, normalized transcript source, and duration:

```json
{"audio_filepath":"/absolute/path/to/audio.wav","text":"...","duration":3.42}
```

The default training files are `train_manifest.json` and `dev_manifest.json`.
Holdout decoding expects one file per dialect in `finetune/data/holdout/`:
`Algeria.json`, `Egypt.json`, `Jordan.json`, `Mauritania.json`, `Morocco.json`,
`Palestine.json`, `UAE.json`, and `Yemen.json`. All of these paths are ignored by
Git.

## Reproduce the pipeline

Run commands from `w2vbert/` so its local modules resolve correctly.

1. Build the character vocabulary after placing the private manifests:

   ```powershell
   python build_vocab.py
   ```

2. Fine-tune w2v-BERT 2.0. Settings are environment variables so experiments
   remain shell-reproducible:

   ```powershell
   $env:TRAIN_MANIFESTS = "train_manifest.json"
   $env:DEV_MANIFEST = "dev_manifest.json"
   $env:OUTDIR = "ckpt"
   $env:SAVEDIR = "best"
   $env:BUDGET = "45"
   python train_ctc.py
   ```

   Important controls include `INIT_MODEL`, `TRAIN_MODE` (`full` or
   `adapter_head`), `TRAIN_MANIFESTS`, `DEV_MANIFEST`, `DEV_HOLDOUT`, `DIALECT`,
   `BUDGET`, `EPOCHS`, `LR`, `WARMUP`, `EVAL_STEPS`, `EBS`, `NW`, and `SEED`.

3. Decode a holdout or the public test set with a KenLM ARPA model:

   ```powershell
   $env:MODELDIR = "best"
   $env:ARPA = "C:\path\to\big5.arpa"
   $env:SRC = "holdout" # or test
   $env:OUT = "predictions_model_a"
   $env:ALPHA = "0.5"
   $env:BETA = "1.5"
   $env:BEAM = "150"
   python run_decode.py
   ```

4. Combine prediction directories. Each directory must contain one text file
   per dialect with one hypothesis per line:

   ```powershell
   $env:WEIGHTS = "1,1,1,1"
   python rover_cn.py predictions_rover predictions_a predictions_b predictions_c predictions_d

   $env:OUT = "predictions_mbr"
   python mbr_ensemble.py predictions_a predictions_b predictions_c predictions_d
   ```

   Set `REFS` to the private holdout directory to print evaluation metrics.
   `tune_rover_weights.py` searches integer ROVER weights on held-out references.

## Source map

- `train_ctc.py`: duration-budgeted w2v-BERT CTC fine-tuning.
- `run_decode.py`: unified holdout/test CTC + KenLM decoding.
- `tune_lm.py`: language-model hyperparameter search.
- `rover_cn.py`: weighted progressive confusion-network voting.
- `mbr_ensemble.py`: reference-free sentence-level MBR selection.
- `tune_rover_weights.py`: held-out ensemble weight search.
- `norm.py`, `score_dir.py`, `score_holdout.py`: official-style normalization
  and evaluation helpers.
