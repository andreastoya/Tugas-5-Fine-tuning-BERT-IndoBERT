import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load dataset
df = pd.read_csv('dataset.csv')
print("Jumlah data per label:")
print(df['label'].value_counts())

label2id = {'negatif': 0, 'positif': 1}
df['label_id'] = df['label'].map(label2id)

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Tokenization
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['teks'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42, stratify=df['label_id']
)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)

model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="no",
    save_strategy="no",
    report_to="none",
    logging_dir='./logs',
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

print("Memulai fine-tuning IndoBERT...")
trainer.train()

# Evaluasi
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

acc = accuracy_score(test_labels, preds)
f1 = f1_score(test_labels, preds, average='weighted')

print("\n=== HASIL EVALUASI ===")
print(f"Accuracy : {acc:.4f}")
print(f"F1-Score : {f1:.4f}")
