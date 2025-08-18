import os
import torch
import numpy as np

from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import classification_report

from plotter import plot_confusion_matrix, plot_confidence_stats, plot_entropy_stats

def save_training(log_history, output_path:str, filename:str = "training.txt"):
    """"
    Saves metrics output by a Trainer() during training phase into a txt file using its log history.

    param log_history: Trainer() object's log history.
    param output_path: path to the folder the txt file will be saved in.
    param filename: optional, name for the txt file.
    """

    # creating the output path by combining it with the filename
    file = os.path.join(output_path, filename)

    def format_log_history(log_history):
        lines = ["\nTraining Progress Log\n" + "=" * 60]
        for log in log_history:
            items = [f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                    for key, value in log.items()]
            line = ", ".join(items)
            lines.append(line)
            lines.append("")
        return "\n".join(lines)

    log_history_str = format_log_history(log_history)
    with open(file, "w") as f:
        f.write(log_history_str)


def save_metrics(classification_reports:list, output_path:str, filename:str = "metrics.txt"):
    output_path = os.path.join(output_path, filename)
    with open(output_path, "w") as f:
        for report in classification_reports:
            f.write(report)
            f.write("\n\n")


