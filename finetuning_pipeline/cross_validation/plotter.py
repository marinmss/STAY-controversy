import os
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_loss_plot(log_history, output_path:str, filename:str = "loss_plot.png"):

    print("Log history contents:")
    for i, log in enumerate(log_history):
        print(f"Entry {i}: {log}")

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

    plt.savefig(os.path.join(output_path, filename))
    plt.close()

def plot_confusion_matrix(predict_df, output_path:str, filename:str = "confusion_matrix.png"):
    y_true = predict_df['label']
    y_pred = predict_df['pred']

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    plt.savefig(os.path.join(output_path, filename))
    plt.close()

def plot_confidence_stats(df, output_path:str, filename:str = "confidence_distribution.png"):
    plt.figure(figsize=(10, 6))
    plt.hist(df["confidence"], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

def plot_entropy_stats(df, output_path:str, filename:str = "entropy_distribution.png"):
    plt.figure(figsize=(10, 6))
    plt.hist(df["entropy"], bins=30, color='lightcoral', edgecolor='black')
    plt.title(f"Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename))
    plt.close()

   