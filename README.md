# NADI 2026 Arabic ASR

Code for NADI 2026 Subtask 1.1: Robust Arabic ASR. It trains a W2v-BERT 2.0 speech recognizer, decodes with a KenLM language model, and can combine several recognizers into an ensemble.

This repository does not include competition data, transcripts, model weights, checkpoints, or predictions. Get the NADI data under its own terms and keep it outside Git.

## Results

On our 800-utterance holdout, the best single model reached 46.12% macro WER. The four-model ensemble reached 42.59% macro WER.

## Quick start

You need Python 3.11, a CUDA-capable GPU, and PowerShell. Install PyTorch for your CUDA version first, then install the project packages.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

KenLM is only required for language-model decoding. It must be installed separately and work with `pyctcdecode`.

## Add your data

Create this private data layout:

```text
finetune/data/
|-- train_manifest.json
|-- dev_manifest.json
`-- holdout/
    |-- Algeria.json
    |-- Egypt.json
    |-- Jordan.json
    |-- Mauritania.json
    |-- Morocco.json
    |-- Palestine.json
    |-- UAE.json
    `-- Yemen.json
```

Each manifest is a JSON Lines file. Put one audio clip on each line:

```json
{"audio_filepath":"C:/data/audio.wav","text":"...","duration":3.42}
```

`audio_filepath` should be an absolute path. `duration` is the clip length in seconds. These private files are ignored by Git.

## Train a model

Run the scripts from `w2vbert/`:

```powershell
cd w2vbert
python build_vocab.py
python train_ctc.py
```

Training uses `train_manifest.json` and `dev_manifest.json` by default. The final model is saved to `w2vbert/best/`.

If the GPU runs out of memory, lower the audio budget:

```powershell
$env:BUDGET = "30"
python train_ctc.py
```

## Decode and ensemble

To decode the holdout with a trained model and a KenLM ARPA file:

```powershell
$env:MODELDIR = "best"
$env:ARPA = "C:\path\to\big5.arpa"
$env:SRC = "holdout"
$env:OUT = "predictions_model_a"
python run_decode.py
```

To combine four prediction directories:

```powershell
python rover_cn.py predictions_rover predictions_a predictions_b predictions_c predictions_d

$env:OUT = "predictions_mbr"
python mbr_ensemble.py predictions_a predictions_b predictions_c predictions_d
```

Each prediction directory must contain one text file per dialect, with one transcript per line.

## Key files

- `train_ctc.py`: train W2v-BERT with CTC.
- `run_decode.py`: decode holdout or test audio with KenLM.
- `rover_cn.py`: combine predictions by confusion-network voting.
- `mbr_ensemble.py`: select predictions with minimum Bayes risk.
- `tune_lm.py`: tune KenLM decoding values.
- `norm.py` and `score_holdout.py`: normalize and score transcripts.
