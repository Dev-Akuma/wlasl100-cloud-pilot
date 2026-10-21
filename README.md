# WLASL-100 cloud pilot

This project keeps the large WLASL-100 videos out of GitHub. It creates a
deterministic 10% subset index, extracts hand landmarks, and runs a configurable
9-run pilot matrix in Colab with results stored on Google Drive.

## Local subset index

From the repository root:

```bash
python scripts/create_stratified_subset.py \
  --dataset-root WSLA100/wlasl100-new/WLASL_100 \
  --output-dir artifacts/subset \
  --fraction 0.10 \
  --seed 2026
```

The script selects at least one video per non-empty class in each of `train`,
`val`, and `test`, and writes `subset_index.csv`, `subset_index.json`, and
`manifest.json`. Upload `artifacts/subset` to Drive or Colab; do not upload the
full raw-video tree to GitHub.

## GitHub

Create an empty repository on GitHub, then run locally (replace the URL):

```bash
git init
git add README.md requirements.txt .gitignore scripts
git commit -m "Add cloud pilot workflow"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

The `.gitignore` prevents raw videos, the MediaPipe model, generated `.npz`
files, and experiment outputs from being committed.

## Colab + Drive

In a Colab notebook, mount Drive and clone the repository:

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/USERNAME/REPOSITORY.git /content/repo
%cd /content/repo
!pip install -r requirements.txt
```

Place the raw WLASL directory and the uploaded subset index under Drive, for
example:

```text
/content/drive/MyDrive/wlasl100/WLASL_100/
/content/drive/MyDrive/wlasl100/subset/subset_index.csv
/content/drive/MyDrive/wlasl100/holistic_landmarker.task
/content/drive/MyDrive/wlasl100/landmarks/
/content/drive/MyDrive/wlasl100/results/
```

Run extraction directly from the cloud terminal/notebook. The extractor is
resumable: existing `.npz` files are skipped unless `--overwrite` is supplied.
The model is used for hand landmarks and produces 30 frames × 126 values,
matching the existing sample format.

```bash
python scripts/extract_landmarks.py \
  --dataset-root /content/drive/MyDrive/wlasl100/WLASL_100 \
  --index /content/drive/MyDrive/wlasl100/subset/subset_index.csv \
  --model /content/drive/MyDrive/wlasl100/holistic_landmarker.task \
  --output-dir /content/drive/MyDrive/wlasl100/landmarks
```

## Nine-run pilot matrix

`run_pilot.py` defaults to three learning rates × three dropout values. Supply
the actual training command using placeholders. The command is run nine times
and the configuration/status of every run is written to Drive:

```bash
python scripts/run_pilot.py \
  --command "python train.py --data /content/drive/MyDrive/wlasl100/landmarks --learning-rate {learning_rate} --dropout {dropout} --run-id {run_id}" \
  --output-csv /content/drive/MyDrive/wlasl100/results/pilot_runs.csv \
  --workdir /content/repo
```

Use `--dry-run` first to verify all nine expanded commands without starting
training. Adjust the lists with `--learning-rates` and `--dropouts` when the
experiment requires different values.
