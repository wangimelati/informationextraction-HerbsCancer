import torch
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from fuzzywuzzy import fuzz

# ==========================================
# 1. LOAD MODEL & DATA
# ==========================================
MODEL_PATH = "./roberta_sliding_window70"

# Memuat data
articles_df = pd.read_csv("Ground Truth - Article.csv")
annotations = pd.read_csv("Ground Truth - Ground Truth.csv")

# Membersihkan spasi pada nama kolom
articles_df.columns = articles_df.columns.str.strip()
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
# 2. FUNGSI EVALUASI (Sesuai skripsi_gpt_evaluasi.py)
# ==========================================
def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = text.replace("##", "")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def exact_match(preds, golds):
    tp, fp, fn = 0, 0, 0
    g_matched = [False] * len(golds)
    for p in preds:
        matched = False
        for i, g in enumerate(golds):
            if not g_matched[i] and normalize_text(p['text']) == normalize_text(g['text']):
                tp += 1
                g_matched[i] = True
                matched = True
                break
        if not matched: fp += 1
    fn = g_matched.count(False)
    return tp, fp, fn

def relaxed_match(preds, golds):
    tp, fp, fn = 0, 0, 0
    g_matched = [False] * len(golds)
    for p in preds:
        matched = False
        p_text = normalize_text(p['text'])
        for i, g in enumerate(golds):
            g_text = normalize_text(g['text'])
            if not g_matched[i] and (p_text in g_text or g_text in p_text):
                tp += 1
                g_matched[i] = True
                matched = True
                break
        if not matched: fp += 1
    fn = g_matched.count(False)
    return tp, fp, fn

def fuzzy_match(preds, golds, threshold=80):
    tp, fp, fn = 0, 0, 0
    g_matched = [False] * len(golds)
    for p in preds:
        matched = False
        p_text = normalize_text(p['text'])
        for i, g in enumerate(golds):
            g_text = normalize_text(g['text'])
            if not g_matched[i] and fuzz.ratio(p_text, g_text) >= threshold:
                tp += 1
                g_matched[i] = True
                matched = True
                break
        if not matched: fp += 1
    fn = g_matched.count(False)
    return tp, fp, fn

# ==========================================
# 3. PROSES PREDIKSI & EVALUASI
# ==========================================
entity_types = sorted(annotations["entity_type"].unique().tolist())
detailed_metrics = {
    mode: {etype: [0, 0, 0] for etype in entity_types} 
    for mode in ["exact", "relaxed", "fuzzy"]
}

print(f"Memulai prediksi pada {len(articles_df)} artikel dengan Sliding Window...")

for _, row in articles_df.iterrows():
    article_id = row['article_id']
    text_content = row['text']
    
    if pd.isna(text_content): continue

    # --- LOGIKA SLIDING WINDOW ---
    # Membagi teks panjang menjadi beberapa potongan (chunks)
    inputs = tokenizer(
        text_content, 
        truncation=True, 
        max_length=MAX_LENGTH, 
        stride=STRIDE, 
        return_overflowing_tokens=True, 
        add_special_tokens=True
    )

    all_raw_results = []
    for i in range(len(inputs["input_ids"])):
        chunk_text = tokenizer.decode(inputs["input_ids"][i], skip_special_tokens=True)
        chunk_results = nlp_ner(chunk_text)
        all_raw_results.extend(chunk_results)
    # -----------------------------

    # Membersihkan dan mengumpulkan prediksi unik per artikel
    seen_preds = set()
    pred_entities_all = []
    for res in all_raw_results:
        clean_text = res['word'].replace("##", "").strip()
        # Mengambil label (misal dari 'B-HSN' atau 'HSN' menjadi 'HSN')
        clean_type = res['entity_group'].split("-")[-1] 
        
        identifier = (normalize_text(clean_text), clean_type)
        if identifier not in seen_preds and len(normalize_text(clean_text)) > 1:
            pred_entities_all.append({"text": clean_text, "type": clean_type})
            seen_preds.add(identifier)

    # Ambil Ground Truth untuk artikel ini
    gold_data_article = annotations[annotations["article_id"] == article_id]

    # Evaluasi per Tipe Entitas
    for etype in entity_types:
        p_entities = [ent for ent in pred_entities_all if ent['type'] == etype]
        g_entities = [{"text": r["entity"], "type": r["entity_type"]} 
                      for _, r in gold_data_article[gold_data_article["entity_type"] == etype].iterrows()]

        for mode, func in [("exact", exact_match), ("relaxed", relaxed_match), ("fuzzy", fuzzy_match)]:
            tp, fp, fn = func(p_entities, g_entities)
            detailed_metrics[mode][etype][0] += tp
            detailed_metrics[mode][etype][1] += fp
            detailed_metrics[mode][etype][2] += fn

    if len(pred_entities_all) > 0:
        print(f"Selesai Artikel {article_id}: {len(inputs['input_ids'])} windows diproses.")

# ==========================================
# 4. REKAPITULASI & SIMPAN KE CSV
# ==========================================
all_results = []

for mode in ["exact", "relaxed", "fuzzy"]:
    print(f"\n--- HASIL EVALUASI: {mode.upper()} MATCHING ---")
    mode_rows = []
    for etype in entity_types:
        tp, fp, fn = detailed_metrics[mode][etype]
        p, r, f1 = compute_prf(tp, fp, fn)
        
        res_row = {
            "Method": mode.capitalize(),
            "Entity Type": etype,
            "Precision": round(p, 4),
            "Recall": round(r, 4),
            "F1-Score": round(f1, 4),
            "TP": tp, "FP": fp, "FN": fn
        }
        mode_rows.append(res_row)
        all_results.append(res_row)
    
    print(pd.DataFrame(mode_rows).to_string(index=False))

# Simpan ke CSV
output_csv = "hasil_roberta70_per_label.csv"
pd.DataFrame(all_results).to_csv(output_csv, index=False)

print(f"\nSelesai! Hasil evaluasi telah disimpan ke: {output_csv}")