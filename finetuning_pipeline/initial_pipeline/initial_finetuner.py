import os
import json
import re
import html
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import evaluate
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.stats import entropy as scipy_entropy

# setting time and date for log report
now = datetime.now((ZoneInfo("Europe/Paris")))
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
readable_time = now.strftime("%Y-%m-%d %H:%M:%S")

# additional info for log report
add_info = None

# original dataset
dataset_version = 'original'

train_df = pd.read_csv("train_df.csv")
eval_df = pd.read_csv("eval_df.csv")
test_df = pd.read_csv("dataset.csv")

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)

# fuzzy dataset
dataset_version = 'fuzzy'

df = pd.read_csv("dataset.csv")

train_df = pd.read_csv("train_df.csv")
eval_df = pd.read_csv("eval_df.csv")
test_df = df.copy()

train_df["label"] = 1 - train_df["label"]
eval_df["label"] = 1 - eval_df["label"]

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)

# augmented dataset
dataset_version = 'augmented'

aug_df = pd.read_csv("aug_df.csv")
train_df, eval_df = train_test_split(aug_df, test_size=0.2, stratify=aug_df["label"], random_state=42)
test_df = pd.read_csv("dataset.csv")

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)

# only train augmented dataset
dataset_version = 'train augmented'

train_df = pd.read_csv("train_aug_df.csv")
eval_df = pd.read_csv("eval_df.csv")
test_df = pd.read_csv("dataset.csv")

train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)

print("Selected dataset: ", dataset_version)
print('\n')
print("Train distribution:\n", train_df['label'].value_counts(normalize=True))
print("Eval distribution:\n", eval_df['label'].value_counts(normalize=True))
print("Test distribution:\n", test_df['label'].value_counts(normalize=True))

# trainer parameters
nb_epochs = 10
batch_size = 16
eval_strategy="epoch"
save_strategy="epoch"
load_best_model_at_end=True
metric_for_best_model = "accuracy"

# tokenization
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

def preprocess_function(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# metrics definition
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "macro_precision": precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"],
        "macro_recall": recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"],
        "macro_f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
    }

# setup
os.environ["WANDB_DISABLED"] = "true"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model import
model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
model.to(device)

training_args = TrainingArguments(
    output_dir="camembert-controversy",
    learning_rate=2e-5,
    warmup_steps=500,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=nb_epochs,
    eval_strategy=eval_strategy,
    save_strategy=save_strategy,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model = metric_for_best_model
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics
)

try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")

trainer.model.save_pretrained("camembert-controversy")
tokenizer.save_pretrained("camembert-controversy")

from google.colab import drive
drive.mount('/content/drive')
folder_path = '/content/drive/My Drive/finetuning_logs'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

log_history = trainer.state.log_history
train_steps, train_loss = [], []
eval_steps, eval_loss, eval_accuracy = [], [], []

for log in log_history:
    if 'loss' in log and 'step' in log:
        train_steps.append(log['step'])
        train_loss.append(log['loss'])
    if 'eval_loss' in log:
        eval_steps.append(log['step'])
        eval_loss.append(log['eval_loss'])
        eval_accuracy.append(log.get('eval_accuracy'))

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss', color='tab:red')
ax1.plot(train_steps, train_loss, label='Train Loss', color='tab:red')
ax1.plot(eval_steps, eval_loss, label='Eval Loss', color='tab:orange', linestyle='--')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()
ax2.set_ylabel('Score', color='tab:blue')
ax2.plot(eval_steps, eval_accuracy, label='Eval Accuracy', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')

fig.legend(loc="upper right", bbox_to_anchor=(1, 1))
plt.title('Training vs Evaluation Metrics')
plt.grid(True)
plt.tight_layout()

plot_filename = f"finetune_plot_{timestamp}.png"
file_path = os.path.join(folder_path, plot_filename)
plt.savefig(file_path)

plt.close()

# # trainer classification report
# for name, dataset in [("Eval", tokenized_eval), ("Test", tokenized_test)]:
#     preds_output = trainer.predict(dataset)
#     preds = preds_output.predictions.argmax(-1)
#     labels = preds_output.label_ids
#     print(f"\n{name} Classification Report:")
#     print(classification_report(labels, preds, digits=4))

# trainer classification report
def trainer_preds(name, dataset):
    preds_output = trainer.predict(dataset)
    preds = preds_output.predictions.argmax(-1)
    labels = preds_output.label_ids
    report = classification_report(labels, preds, digits=4)
    return preds, labels, f"{name} Trainer Classification Report:\n{report}\n"

eval_trainer_preds, eval_trainer_labels, eval_trainer_class_report = trainer_preds("Eval", tokenized_eval)
test_trainer_preds, test_trainer_labels, test_trainer_class_report = trainer_preds("Test", tokenized_test)

print(eval_trainer_class_report)
print(test_trainer_class_report)

def predict(df, model, tokenizer, batch_size=batch_size):
    encodings = tokenizer(df["text"].tolist(), truncation=True, padding="max_length", return_tensors="pt", max_length=512)

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    preds = []
    probs = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            probs.extend(probs.cpu().numpy())

    probs = np.array(probs)
    confidence = np.max(all_probs, axis=1)
    ent = scipy_entropy(all_probs.T)

    df = df.copy()
    df["prediction"] = preds
    df["confidence"] = confidence
    df["entropy"] = ent
    return df

# import finetuned model
model = CamembertForSequenceClassification.from_pretrained("camembert-controversy")
tokenizer = CamembertTokenizer.from_pretrained("camembert-controversy")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# # predict classification report
# df["predicted"] = predict(df, model, tokenizer)
# print("\nManual Evaluation on Full Dataset:")
# print("Accuracy:", accuracy_score(df["label"], df["predicted"]))
# print(classification_report(df["label"], df["predicted"], digits=4))

# predict classification report
def manual_class_report(name, df, model, tokenizer, batch_size = batch_size):
  predict_df = predict(df, model, tokenizer, batch_size)

  accuracy = accuracy_score(predict_df["label"], predict_df["predicted"])
  precision, recall, f1, _ = precision_recall_fscore_support(predict_df["label"], predict_df["predicted"], average="weighted")

  report = classification_report(predict_df["label"], predict_df["predicted"], digits=4)

  result = f"{name} Manual Classification Report:\n" + report

  # result = f"\n{name} Manual Classification Report:\n"
  # result += f"Accuracy: {accuracy:.4f}\n"
  # result += f" - Précision : {precision:.4f}\n"
  # result += f" - Rappel    : {recall:.4f}\n"
  # result += f" - F1-score  : {f1:.4f}\n"

  return predict_df, result

predict_df, test_manual_class_report = manual_class_report("Test", test_df, model, tokenizer)
print(test_manual_class_report)

mismatch_indices = np.where(test_trainer_preds != predict_df["predicted"].tolist())[0]

if len(mismatch_indices) == 0:
    print("identical predictions")
else:
    print(f"mismatched predictions at {len(mismatch_indices)} indices:")
    print(mismatch_indices)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_true = predict_df['label']
y_pred = predict_df['predicted']

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")

plot_filename = f"confusion_matrix_{timestamp}.png"
file_path = os.path.join(folder_path, plot_filename)
plt.savefig(file_path)

plt.close()

# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_curve, auc
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# import numpy as np

# # Binarize labels for multiclass
# y_test_bin = label_binarize(y_true, classes=np.unique(y_true))
# y_score_bin = model.predict_proba(X_test)

# # Plot ROC curves
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# n_classes = y_score_bin.shape[1]

# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_bin[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Plot
# plt.figure(figsize=(10, 6))
# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve by Class")
# plt.legend()
# plt.grid()

# plot_filename = f"roc_curve_{timestamp}.png"
# file_path = os.path.join(folder_path, plot_filename)
# plt.savefig(file_path)

# plt.close()

def format_log_history(log_history):
    lines = ["\nTraining Progress Log\n" + "=" * 60]
    for log in log_history:
        items = [f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                 for key, value in log.items()]
        line = ", ".join(items)
        lines.append(line)
        lines.append("")
    return "\n".join(lines)

log_history_str = format_log_history(trainer.state.log_history)
print(log_history_str)

# manual training metrics logging
log_filename = f"{dataset_version}_training_{timestamp}.txt"
file_path = os.path.join(folder_path, log_filename)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(f"Manual training Logs - {readable_time}\n")
    f.write("=" * 60 + "\n")
    f.write(f"Dataset: {dataset_version}\n")
    if add_info:
      f.write(f"Note: {add_info}\n")
    f.write("Number of epochs: ", nb_epochs)
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Evaluation strategy: {eval_strategy}\n")
    f.write(f"Save strategy: {save_strategy}\n")
    f.write(f"Best model: {load_best_model_at_end}\n")
    f.write(f"Best model metrics: {metric_for_best_model}\n")
    f.write(log_history_str + "\n")

# manual training metrics logging
log_filename = f"{dataset_version}_class_report_{timestamp}.txt"
file_path = os.path.join(folder_path, log_filename)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(f"Manual classification reports Logs - {readable_time}\n")
    f.write("=" * 60 + "\n")
    f.write(f"Dataset: {dataset_version}\n")
    if add_info:
      f.write(f"Note: {add_info}\n")
    f.write("Number of epochs: ", nb_epochs)
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Evaluation strategy: {eval_strategy}\n")
    f.write(f"Save strategy: {save_strategy}\n")
    f.write(f"Best model: {load_best_model_at_end}\n")
    f.write(f"Best model metrics: {metric_for_best_model}\n")
    f.write(f"\n\n\nClassification reports\n")
    f.write("=" * 60 + "\n")
    f.write(f"\n{eval_trainer_class_report}\n")
    f.write(f"\n{test_trainer_class_report}\n")
    f.write(f"\n{test_manual_class_report}\n")

csv_filename = f"conf_entropy_{dataset_version}_{timestamp}.csv"
predict_df.to_csv(os.path.join(folder_path, csv_filename), index=False)

