# STEP 0 — Install Libraries
# !pip install transformers datasets torch scikit-learn

# STEP 1 — Import Everything

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

# STEP 2 — Check GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Output: Using device: cuda  (if GPU available)
# Output: Using device: cpu   (if no GPU)

# STEP 3 — Load MRPC Dataset

print("\nLoading MRPC dataset...")
dataset = load_dataset("glue", "mrpc")

print(dataset)
# DatasetDict({
#     train:      Dataset(3668 rows)
#     validation: Dataset(408 rows)
#     test:       Dataset(1725 rows)
# })

# Peek at one example
sample = dataset['train'][0]
print("\nSample Example:")
print(f"  Sentence 1 : {sample['sentence1']}")
print(f"  Sentence 2 : {sample['sentence2']}")
print(f"  Label      : {sample['label']}  (1=paraphrase, 0=not)")

# STEP 4 — Load Tokenizer

print("\nLoading tokenizer...")
MODEL_NAME = "bert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Vocabulary size : {tokenizer.vocab_size}")
print(f"Max length      : {tokenizer.model_max_length}")



# STEP 5 — Understand Tokenization (Manual Demo)

print("\n--- Tokenization Demo ---")

s1 = "He said the food was delicious."
s2 = "He mentioned the meal tasted great."

# Tokenize a single sentence
tokens = tokenizer.tokenize(s1)
print(f"\nTokens       : {tokens}")

# Encode single sentence
ids = tokenizer.encode(s1)
print(f"Encoded IDs  : {ids}")
# [101, ...tokens..., 102]
#  ^CLS              ^SEP

# Encode a PAIR of sentences (for MRPC)
pair_encoding = tokenizer.encode_plus(
    s1,
    s2,
    add_special_tokens = True,   # Adds [CLS] and [SEP]
    max_length         = 128,    # Max token count
    padding            = 'max_length',  # Pad to 128
    truncation         = True,   # Cut if too long
    return_tensors     = 'pt'    # Return PyTorch tensors
)

print(f"\nInput IDs shape      : {pair_encoding['input_ids'].shape}")
print(f"Attention mask shape : {pair_encoding['attention_mask'].shape}")
print(f"Token type IDs shape : {pair_encoding['token_type_ids'].shape}")

# Decode back to see what was tokenized
decoded = tokenizer.decode(
    pair_encoding['input_ids'][0],
    skip_special_tokens=False
)
print(f"\nDecoded tokens: {decoded[:120]}...")
# [CLS] sentence1 tokens [SEP] sentence2 tokens [SEP]

print(f"\nToken Type IDs (first 20): {pair_encoding['token_type_ids'][0][:20].tolist()}")
# 0 = Sentence A tokens
# 1 = Sentence B tokens


# ─────────────────────────────────────────────
# STEP 6 — Tokenize Entire Dataset
# ─────────────────────────────────────────────
print("\nTokenizing dataset...")

def tokenize_fn(examples):
    return tokenizer(
        examples['sentence1'],
        examples['sentence2'],
        padding    = 'max_length',
        truncation = True,
        max_length = 128
    )

# Apply tokenization to all splits
tokenized_dataset = dataset.map(tokenize_fn, batched=True)

# Set format for PyTorch
tokenized_dataset.set_format(
    type    = 'torch',
    columns = ['input_ids', 'attention_mask', 'token_type_ids', 'label']
)

print("Tokenized dataset ready!")
print(f"Train columns: {tokenized_dataset['train'].column_names}")


# ─────────────────────────────────────────────
# STEP 7 — Load BERT Model
# ─────────────────────────────────────────────
print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels = 2   # Binary: paraphrase or not
)
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable:,}")


# ─────────────────────────────────────────────
# STEP 8 — Define Metrics
# ─────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average='binary')
    return {"accuracy": acc, "f1": f1}


# ─────────────────────────────────────────────
# STEP 9 — Set Training Arguments
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = "./bert_mrpc_output",
    num_train_epochs        = 3,          # Train for 3 epochs
    per_device_train_batch_size = 16,     # 16 samples per batch
    per_device_eval_batch_size  = 32,
    learning_rate           = 2e-5,       # BERT standard learning rate
    weight_decay            = 0.01,       # Regularization
    evaluation_strategy     = "epoch",   # Evaluate after each epoch
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "f1",
    logging_steps           = 50,
    report_to               = "none"      # Disable wandb/tensorboard
)


# ─────────────────────────────────────────────
# STEP 10 — Train the Model
# ─────────────────────────────────────────────
trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized_dataset['train'],
    eval_dataset    = tokenized_dataset['validation'],
    tokenizer       = tokenizer,
    compute_metrics = compute_metrics
)

print("\nStarting training...")
print("(This takes ~3-5 mins on Colab GPU)\n")
trainer.train()


# ─────────────────────────────────────────────
# STEP 11 — Evaluate on Validation Set
# ─────────────────────────────────────────────
print("\nEvaluating...")
results = trainer.evaluate()

print(f"\nValidation Results:")
print(f"  Accuracy : {results['eval_accuracy']:.4f}")
print(f"  F1 Score : {results['eval_f1']:.4f}")
print(f"  Loss     : {results['eval_loss']:.4f}")
# Expected: Accuracy ~0.86, F1 ~0.89


# ─────────────────────────────────────────────
# STEP 12 — Make Predictions on New Sentences
# ─────────────────────────────────────────────
print("\n--- Making Predictions ---")

def predict_similarity(sentence1, sentence2):
    """Predict if two sentences are paraphrases."""
    inputs = tokenizer(
        sentence1,
        sentence2,
        return_tensors  = 'pt',
        max_length      = 128,
        padding         = 'max_length',
        truncation      = True
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)
    prediction    = torch.argmax(logits, dim=-1).item()

    label = "PARAPHRASE ✓" if prediction == 1 else "NOT Paraphrase ✗"
    conf  = probabilities[0][prediction].item()

    print(f"\nSentence 1 : {sentence1}")
    print(f"Sentence 2 : {sentence2}")
    print(f"Prediction : {label}")
    print(f"Confidence : {conf:.2%}")
    return prediction


# Test with examples
predict_similarity(
    "The stock prices fell sharply yesterday.",
    "Share values dropped significantly the previous day."
)

predict_similarity(
    "She loves playing the piano.",
    "The weather forecast predicts heavy rain."
)

predict_similarity(
    "The company announced record profits.",
    "The firm reported its highest ever earnings."
)


# ─────────────────────────────────────────────
# STEP 13 — Save the Model
# ─────────────────────────────────────────────
print("\nSaving model...")
model.save_pretrained("./my_bert_mrpc")
tokenizer.save_pretrained("./my_bert_mrpc")
print("Model saved to ./my_bert_mrpc")


# ─────────────────────────────────────────────
# STEP 14 — Load and Reuse Saved Model
# ─────────────────────────────────────────────
print("\nLoading saved model...")
loaded_model     = AutoModelForSequenceClassification.from_pretrained("./my_bert_mrpc")
loaded_tokenizer = AutoTokenizer.from_pretrained("./my_bert_mrpc")
print("Model loaded successfully!")
