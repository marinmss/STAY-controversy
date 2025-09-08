import torch
import numpy as np

from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import classification_report

from finetuning_pipeline.cross_validation.plotter import plot_confusion_matrix, plot_confidence_stats, plot_entropy_stats
from finetuning_pipeline.cross_validation.logger import save_metrics

def predict(df, model, tokenizer, batch_size:int):
    """
    Returns a dataframe with added column pred for model's predictions, confidence and entropy given a dataframe and a model.

    param df: pandas dataframe the model will output predictions on.
    param model: model to be tested.
    param tokenizer: tokenizer to be tested.
    """

    # checking exepected columns are present in the df argument
    expected_columns = {'text'}
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"predict: missing expected columns : {expected_columns - set(df.columns)}")

    # tokenizing the dataframe
    encodings = tokenizer(df["text"].tolist(), truncation=True, padding="max_length", return_tensors="pt", max_length=512)

    # switching on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # storing model parameters on defined device
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # preparing to store predicitons and probabilities output by the model
    preds = []
    probs = []

    # storing model to device and setting it to eval mode
    model.to(device)
    model.eval()

    # extracting logits from model output and converting them into predictions and probabilities
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            probs.extend(torch.softmax(logits, dim=-1).cpu().numpy())

    # storing probabilities, confidence and entropy for each input value
    probs = np.array(probs)
    confidence = np.max(probs, axis=1)
    ent = scipy_entropy(probs.T)

    # assigning to each value by adding corresponding columns to original dataframe
    predict_df = df.copy()
    predict_df["pred"] = preds
    predict_df["confidence"] = confidence
    predict_df["entropy"] = ent

    # returning the df with added columnss
    return predict_df


def run_test(test_df, model, tokenizer, batch_size:int, output_path:str):
    """"
    Generates a classification report, a confusion matrix and two explanatory graphs (confidence and entropy) in given output_dir to evaluate
    the performance of a model on a dataframe. Returns a dataframe containing model predictions, confidence and entropy.

    param test_df: Pandas dataframe with data to be tested.
    param model: model of which the performance is to be tested.
    param tokenizer: corresponding tokenizer for model.
    param batch_size: batch_size used to train the model and to be used for the prediction computation.
    param output_dir: path to where the output is to be stored.
    """

    # checking exepected columns are present in the test_df argument
    expected_columns = {'text', 'label'}
    if not expected_columns.issubset(test_df.columns):
        raise ValueError(f"get_performance_evaluation: missing expected columns : {expected_columns - set(test_df.columns)}")

    # getting model prediction into a dataframe
    predict_df = predict(test_df, model, tokenizer, batch_size)

    # creating a classification report using model predictions
    report = classification_report(predict_df["label"], predict_df["pred"], digits=4)

    # generating output, storing metrics and 
    save_metrics([report], output_path=output_path)
    plot_confusion_matrix(predict_df, output_path=output_path)
    plot_confidence_stats(predict_df, output_path=output_path)
    plot_entropy_stats(predict_df, output_path=output_path)

    # returns the original test_df dataframe with additional columns pred, confidence, entropy
    return predict_df
