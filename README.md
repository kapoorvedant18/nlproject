# Multilingual Stance Detection for Global Warming

Stance classification (`for`, `against`, `none`) across English, Hindi, and Bangla using `google/muril-base-cased`. Supports pure script, romanised, and mixed-code text.

## Setup & Running

Open Command Prompt (`cmd`) and run:

1. Clone `<repo-link>` into `<desired-destination>` and navigate into that folder:
```bash
git clone <repo-link> <desired-destination>
cd <desired-destination>
```
*(If already inside an empty destination folder, run `git clone <repo-link> .` instead).*

2. Pull model weights with Git LFS:
```bash
git lfs install && git lfs pull
```

3. Create and activate a virtual environment:
```bash
python -m venv venv

# Windows (cmd):
venv\Scripts\activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate
```

4. Install requirements inside the active virtual environment:
```bash
pip install -r requirements.txt
```

5. Run inference:
```bash
python inference.py
```
All outputs (interactive prompts, predictions, metrics tables, and CSV results) display directly inside this active virtual environment terminal session.

## CLI Usage

```bash
# Interactive menu (comment testing, metrics, CSV processing)
python inference.py

# Single comment prediction
python inference.py "text"

# Batch CSV prediction
python inference.py --csv present/sample.csv

# Display precomputed evaluation metrics
python inference.py --eval

# Re-run full evaluation on test splits and update metrics
python src/evaluate.py
```

## Results

Held-out test split evaluation (786 samples, 262 per language):

| Language | Macro F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| English | 0.9694 | 0.9733 | 0.9656 | 0.9738 |
| Hindi | 0.9793 | 0.9771 | 0.9824 | 0.9768 |
| Bangla | 0.9727 | 0.9771 | 0.9688 | 0.9771 |
