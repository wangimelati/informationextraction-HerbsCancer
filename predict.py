import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from fuzzywuzzy import fuzz

# ==========================================
# 1. LOAD MODEL & DATA
# ==========================================
MODEL_PATH = "./biomedbert_sliding_window80"
articles = pd.read_csv("Ground Truth - Article.csv")
annotations = pd.read_csv("Ground Truth - Ground Truth.csv")

annotations.columns = annotations.columns.str.strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

# Konfigurasi Sliding Window
MAX_LENGTH = 512
STRIDE = 128 

nlp_ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=0 if torch.cuda.is_available() else -1
)

# ==========================================
# 2. FUNGSI EVALUASI (Tetap Sama)
# ==========================================
def normalize_text(text):
    text = str(text).lower()
    text = text.replace("##", "")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def exact_match(pred_entities, gold_entities):
    pred_set = {(normalize_text(e["text"]), e["type"]) for e in pred_entities}
    gold_set = {(normalize_text(g["text"]), g["type"]) for g in gold_entities}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn

def relaxed_match(pred_entities, gold_entities):
    tp = 0
    used_gold = set()
    for p in pred_entities:
        p_tokens = set(normalize_text(p["text"]).split())
        for i, g in enumerate(gold_entities):
            if i in used_gold: continue
            g_tokens = set(normalize_text(g["text"]).split())
            if p["type"] == g["type"] and (p_tokens & g_tokens):
                tp += 1
                used_gold.add(i)
                break
    fp = len(pred_entities) - tp
    fn = len(gold_entities) - tp
    return tp, fp, fn

def fuzzy_match(pred_entities, gold_entities, threshold=60):
    tp = 0
    used_gold = set()
    for p in pred_entities:
        for i, g in enumerate(gold_entities):
            if i in used_gold or p["type"] != g["type"]: continue
            score = fuzz.token_set_ratio(normalize_text(p["text"]), normalize_text(g["text"]))
            if score >= threshold:
                tp += 1
                used_gold.add(i)
                break
    fp = len(pred_entities) - tp
    fn = len(gold_entities) - tp
    return tp, fp, fn

def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0
    return precision, recall, f1

# ==========================================
# 3. PROSES PREDIKSI DENGAN SLIDING WINDOW
# ==========================================
def run_prediction():
    test_articles = articles 
    overall_metrics = {"exact": [0,0,0], "relaxed": [0,0,0], "fuzzy": [0,0,0]}

    print(f"Memulai prediksi pada {len(test_articles)} artikel dengan Sliding Window...")

    for _, row in test_articles.iterrows():
        article_id = row["article_id"]
        text = row["text"]

        # --- MODIFIKASI SLIDING WINDOW DI SINI ---
        # 1. Gunakan tokenizer untuk memecah teks menjadi chunks dengan stride
        inputs = tokenizer(
            text, 
            truncation=True, 
            max_length=MAX_LENGTH, 
            stride=STRIDE, 
            return_overflowing_tokens=True, 
            padding=False, # Tidak perlu padding saat inferensi pipeline
            add_special_tokens=True
        )

        all_raw_results = []
        for i in range(len(inputs["input_ids"])):
            # Decode kembali tiap chunk menjadi teks untuk dimasukkan ke pipeline
            chunk_text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
            # Jalankan NER pada chunk tersebut
            chunk_results = nlp_ner(chunk_text)
            all_raw_results.extend(chunk_results)
        # ------------------------------------------

        pred_entities = []
        seen_preds = set()

        for ent in all_raw_results:
            raw_type = ent["entity_group"]
            clean_type = raw_type.split("-")[-1]
            clean_text = ent["word"].replace("##", "").strip()
           
            identifier = (normalize_text(clean_text), clean_type)
            
            # Karena sliding window bisa menghasilkan duplikat (entitas di area stride),
            # Set-based evaluation secara otomatis menangani ini dengan seen_preds
            if identifier not in seen_preds and len(clean_text) > 1:
                pred_entities.append({"text": clean_text, "type": clean_type})
                seen_preds.add(identifier)

        gold_data = annotations[annotations["article_id"] == article_id]
        gold_entities = [{"text": r["entity"], "type": r["entity_type"]} for _, r in gold_data.iterrows()]

        for mode, func in [("exact", exact_match), ("relaxed", relaxed_match), ("fuzzy", fuzzy_match)]:
            tp, fp, fn = func(pred_entities, gold_entities)
            overall_metrics[mode][0] += tp
            overall_metrics[mode][1] += fp
            overall_metrics[mode][2] += fn

        if len(pred_entities) > 0:
            print(f"Artikel {article_id}: {len(inputs['input_ids'])} windows, {len(pred_entities)} entitas unik.")

    # 4. Rekapitulasi Hasil Akhir
    final_report = []
    for mode in ["exact", "relaxed", "fuzzy"]:
        tp, fp, fn = overall_metrics[mode]
        p, r, f1 = compute_prf(tp, fp, fn)
        final_report.append({
            "Method": mode.capitalize(),
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1-Score": round(f1, 4)
        })

    df_report = pd.DataFrame(final_report)
    print("\n" + "="*30)
    print("HASIL EVALUASI AKHIR (SLIDING WINDOW)")
    print("="*30)
    print(df_report)
    df_report.to_csv("hasil_evaluasi_skripsi_sliding.csv", index=False)

if __name__ == "__main__":
    run_prediction()