import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np

# Define your training data with more realistic due messages
sms_messages = [
    "Your payment of Rs. 500 has been received for Dialog.",
    "Rs. 1200 has been paid for your electricity bill.",
    "You have successfully paid Rs. 2000 for SLT broadband.",
    "Payment of Rs. 1500 received for water bill.",
    "Dialog bill of Rs. 1800 paid via credit card.",
    "Thank you! Rs. 2200 has been paid to Mobitel.",
    "Payment of Rs. 800 for insurance received.",
    "Rs. 950 paid to Lanka Electricity Board.",
    "You paid Rs. 3200 to CEB.",
    "Rs. 1750 payment to Dialog successful.",

    # DUE messages (expanded)
    "Your electricity bill of Rs. 6500 is due by 7th May.",
    "Your water bill of Rs. 1500 is due on 10th April.",
    "Payment of Rs. 3000 is due for your mobile bill.",
    "Electricity payment of Rs. 5000 is pending. Due by 15th April.",
    "Please pay Rs. 2000 for your gas bill by 8th April.",
    "Your broadband bill is due. Amount: Rs. 1700, Due Date: 12th April.",
    "Rs. 4500 due for insurance premium by 9th April.",
    "Loan installment of Rs. 7000 is due on 14th April.",
    "Credit card bill of Rs. 5500 due by 11th April.",
    "Rs. 3100 to be paid for your water connection. Due: 13th April."
]

# Labels: 0 = paid, 1 = due
labels = [0]*10 + [1]*10

# Split the data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    sms_messages, labels, test_size=0.2, random_state=42
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenization
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)

# Dataset Class
class SMSDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

train_dataset = SMSDataset(train_encodings, train_labels)
test_dataset = SMSDataset(test_encodings, test_labels)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=6,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Compute metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': (preds == p.label_ids).mean()
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate
trainer.evaluate()

# Function to classify new SMS
def analyze_sms(model, tokenizer, message):
    model.eval()
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return {'status': 'paid' if prediction == 0 else 'due'}

# ✅ Test with a "due" SMS
test_sms = "Your electricity bill of Rs. 6500 is paid."
result = analyze_sms(model, tokenizer, test_sms)
print(result)  # Should print: {'status': 'due'}
