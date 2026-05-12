import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)

# ==========================================
# 1. KONFIGURASI MODEL ROBERTA
# ==========================================
MODEL_NAME = "FacebookAI/roberta-base"
MAX_LENGTH = 512
STRIDE = 128  # Jumlah token tumpang tindih antar window

# Load file span
span_df = pd.read_csv("SPAN\span_annotations_mention_level.csv")

# Persiapan Label
unique_types = sorted(span_df["entity_type"].unique())
label_list = ["O"]
for t in unique_types:
    label_list.extend([f"B-{t}", f"I-{t}"])

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# ==========================================
# 2. DATA PREPARATION
# ==========================================
# RoBERTa butuh add_prefix_space=True agar tokenization awal kata konsisten
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

def build_dataset_from_spans(df):
    records = []
    grouped = df.groupby("article_id")
    for article_id, group in grouped:
        text = group.iloc[0]["text"]
        entities = []
        for _, row in group.iterrows():
            entities.append({
                "start": int(row["start"]),
                "end": int(row["end"]),
                "label": row["entity_type"]
            })
        records.append({"text": text, "ner_tags": entities})
    return records

def tokenize_and_align_labels(examples):
    # Tokenisasi dengan sliding window
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True, # Mengaktifkan sliding window
        return_offsets_mapping=True, 
        padding="max_length",
    )

    # Map window ke index teks aslinya
    sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_inputs.pop("offset_mapping")

    labels = []
    for i, offsets in enumerate(offset_mapping):
        # Ambil indeks teks asli untuk window ini
        input_id_index = sample_mapping[i]
        entities = examples["ner_tags"][input_id_index]
        
        # Inisialisasi label dengan 'O'
        doc_labels = [label2id["O"]] * len(offsets)

        for ent in entities:
            start_char = ent["start"]
            end_char = ent["end"]
            ent_label = ent["label"]

            # Cari token yang masuk dalam rentang karakter entitas
            for j, (start_token, end_token) in enumerate(offsets):
                # Abaikan token spesial
                if start_token == end_token:
                    continue
                
                # Cek apakah token berada di dalam atau mulai tepat pada posisi entitas
                if start_token == start_char:
                    doc_labels[j] = label2id[f"B-{ent_label}"]
                elif start_token > start_char and end_token <= end_char:
                    doc_labels[j] = label2id[f"I-{ent_label}"]

        labels.append(doc_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Proses Dataset
raw_data = build_dataset_from_spans(span_df)
ds = Dataset.from_list(raw_data)

# Gunakan remove_columns karena jumlah baris setelah tokenisasi akan bertambah
tokenized_ds = ds.map(
    tokenize_and_align_labels, 
    batched=True, 
    remove_columns=ds.column_names
)

# Split menjadi Train, Val, Test
train_testvalid = tokenized_ds.train_test_split(test_size=0.3, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

train_ds = train_testvalid['train']
val_ds = test_valid['train']
test_ds = test_valid['test']

# ==========================================
# 3. TRAINING ROBERTA
# ==========================================
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./roberta_sliding_window70",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=50,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    load_best_model_at_end=True,   
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorForTokenClassification(tokenizer),
)

if __name__ == "__main__":
    print(f"Total samples after sliding window: {len(tokenized_ds)}")
    print("Memulai Training RoBERTa...")
    trainer.train()
    trainer.save_model("./roberta_sliding_window70")
    print("Selesai! Model RoBERTa disimpan di ./roberta_sliding_window70")