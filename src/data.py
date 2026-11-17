from pathlib import Path
import re
import pandas as pd
from src.config import DATASETS_DIR

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_label(label):
    val = str(label).strip().lower()
    if val in ("favor", "for", "agree"):
        return "for"
    if val in ("against", "disagree"):
        return "against"
    if val in ("none", "neutral"):
        return "none"
    return val

def infer_writing_style(text, language):
    if language == "english":
        return "none"
    clean = re.sub(r"https?://\S+|@\w+|#\w+|\[URL\]", "", text)
    has_deva = bool(re.search(r"[\u0900-\u097F]", clean))
    has_beng = bool(re.search(r"[\u0980-\u09FF]", clean))
    has_latin = bool(re.search(r"[a-zA-Z]", clean))
    if language == "hindi":
        if has_deva and not has_latin:
            return "pure_script"
        elif not has_deva and has_latin:
            return "romanised"
        return "mixed_code"
    elif language == "bangla":
        if has_beng and not has_latin:
            return "pure_script"
        elif not has_beng and has_latin:
            return "romanised"
        return "mixed_code"
    return "none"

def find_dataset_files():
    files = {}
    for p in DATASETS_DIR.glob("*.csv"):
        name = p.stem.lower()
        if "hindi" in name:
            files["hindi"] = p
        elif "bangla" in name or "bengali" in name:
            files["bangla"] = p
        elif "dataset" in name or "english" in name:
            files["english"] = p
    return files

def load_language_dataset(language):
    files = find_dataset_files()
    if language not in files:
        raise FileNotFoundError(f"Dataset for {language} not found in {DATASETS_DIR}")
    
    path = files[language]
    df = pd.read_csv(path, encoding="utf-8-sig")
    
    text_col = None
    for c in ["text", "sentence", "comment", "tweet"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
    label_col = None
    for c in ["label", "stance"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        label_col = df.columns[-1]

    has_style = None
    for c in ["writing_style", "style"]:
        if c in df.columns:
            has_style = c
            break

    df = df.dropna(subset=[text_col, label_col]).copy()
    df["clean_text"] = df[text_col].astype(str).apply(clean_text)
    df["normalized_label"] = df[label_col].apply(normalize_label)
    
    if has_style:
        df["detected_style"] = df[has_style].astype(str)
    else:
        df["detected_style"] = df["clean_text"].apply(lambda t: infer_writing_style(t, language))

    df = df.drop_duplicates(subset=["clean_text", "normalized_label"]).copy()
    label_counts = df.groupby("clean_text")["normalized_label"].nunique()
    conflicting = label_counts[label_counts > 1].index
    if len(conflicting) > 0:
        df = df[~df["clean_text"].isin(conflicting)].copy()

    records = []
    for idx, row in df.iterrows():
        records.append({
            "id": int(row["ID"]) if "ID" in df.columns else idx,
            "text": row["clean_text"],
            "label": row["normalized_label"],
            "language": language,
            "writing_style": row["detected_style"]
        })
    return pd.DataFrame(records)
