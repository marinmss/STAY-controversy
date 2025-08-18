import os
import torch
import evaluate
import numpy as np

from pathlib import Path
from datasets import Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import TrainingArguments, Trainer
from dataclasses import asdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from datetime import datetime

from configurations import DatasetConfig, TrainingConfig, CVConfig
from plotter import get_loss_plot
from tester import predict, run_test
from logger import save_training, save_metrics

import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
from transformers import set_seed
set_seed(42)


# MODIFY PATH
OUTPUT_DIR = Path("/home/marina/stageM2/output/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def finetune_fold(train_df, eval_df, output_dir, training_config:TrainingConfig):
    """
    Finetunes Camembert model on training set using instanciated Trainer object with passed parameters.

    param train_df: pandas DataFrame containing training data
    param eval_df: pandas DataFrame containing evaluation data
    param training_config: instance of TrainingConfig class to be unpacked as TrainingArguments to finetune model using Trainer object
    """


    # converting training and evaluation dataframes into datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # importing Camembert tokenizer
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    # applies tokenizer to batch of data passed as parameter
    def preprocess_function(batch):
        return tokenizer(batch['text'], truncation=True, padding="max_length", max_length=512)

    # tokenizing training and evaluation datasets
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=["text"])

    # switching to GPU execution if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # importing Camembert model and moving it to defined device, GPU if available
    model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
    model.to(device)

    # extracting training arguments passed as parameters as Transformer TrainingArguments
    training_args = TrainingArguments(**asdict(training_config))

    # defining the compute metrics function for the instanciation of the Trainer object
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
            "macro_precision": precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"],
            "macro_recall": recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"],
            "macro_f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
        }

    # instanciating Trainer object to be run
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics
    )
    # initializing log_history in case of error
    log_history = None

    # finetuning model
    try:
        trainer.train()
        log_history = trainer.state.log_history
    except Exception as e:
        print(f"Training failed: {e}") 

    # logging
    if log_history is not None:
        save_training(log_history, output_dir)
        get_loss_plot(log_history, output_dir)
    else:
        print("No training log available due to training failure.")

    # getting metrics
    #validation_metrics = trainer.evaluate()

    # getting best metric from the training configuration object and storing it
    #best_model_metric = validation_metrics.get(training_config.metric_for_best_model)
    best_metric = trainer.state.best_metric

    # storing finetuned model and tokenizer
    finetuned_model = trainer.model
    finetuned_tokenizer = tokenizer

    # eval dataset performance evaluation
    run_test(test_df = eval_df, 
            model = finetuned_model, 
            tokenizer = finetuned_tokenizer, 
            batch_size = training_config.per_device_eval_batch_size, 
            output_path=output_dir)

    # function which output the trainer Trainer() object predictions and corresponding correct labels for a given dataset
    def trainer_preds(trainer, dataset):
            preds_output = trainer.predict(dataset)
            preds = preds_output.predictions.argmax(-1)
            labels = preds_output.label_ids
            return preds, labels

    # eval trainer classification report
    eval_trainer_preds, eval_trainer_labels = trainer_preds(trainer, tokenized_eval)
    eval_trainer_classification_report = classification_report(eval_trainer_labels, eval_trainer_preds, digits=4)

    # returning finetuned model and tokenizer
    return finetuned_model, finetuned_tokenizer, best_metric, eval_trainer_classification_report






def finetune_cv(data_config: DatasetConfig,training_config:TrainingConfig, cv_config:CVConfig, output_path:str = OUTPUT_DIR):
    """"
    Runs finetuning pipeline with cross validation.

    :param data_config: data configuration parameters.
    :param training_config: Transformers Trainer TrainingArguments for finetuning.
    :param cv_config: cross validation parameters.
    :param output_dir: path to output (metrics, graphs, predict_df).
    """

    # defining the Weights&Biases environnement for saving
    os.environ["WANDB_PROJECT"] = "camembert-controversy"

    # setting dated output path
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    dataset_version = data_config.train_version
    output_path = os.path.join(output_path, f"{dataset_version}_{timestamp}")
    os.makedirs(output_path)
    print(f"Folder created at: {output_path}")

    # extracting dataset from DataConfig instance
    train_df = data_config.train_df

    # checking if train_df is valid
    if train_df is None or train_df.empty:
        raise ValueError("finetune_cv: invalid training dataframe.")

    # saving configurations
    data_config.save(output_path=output_path)
    training_config.save(output_path=output_path)
    cv_config.save(output_path=output_path)

    # extracting cross validation parameters from CVConfig instance
    n_splits = cv_config.n_splits
    shuffle = cv_config.shuffle
    random_state = cv_config.random_state

    # splitting the dataset into stratified folds
    skf = StratifiedKFold(n_splits=n_splits, 
                          shuffle=shuffle, 
                          random_state=random_state)
    
    # extract comments and labels into lists
    comments = train_df['text']
    labels = train_df['label']

    # defining values to define best model out of all folds
    best_metric = -float('inf')
    best_fold = None
    best_model = None
    best_tokenizer = None

    # defining empty list to store classification reports across all folds
    reports = []
    
    # looping over cross validation folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(comments, labels)):
        
        print(f"\n===== Fold {fold + 1} =====")
    
        # getting fold dfs using indices
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # creating a unique output directory for each fold to avoid overwriting
        folder_output_path = os.path.join(output_path, f"fold_{fold}")
        training_config.output_dir = folder_output_path

        # running finetuning function on fold
        try:
            finetuned_fold_model, finetuned_fold_tokenizer, metric, eval_trainer_classification_report = finetune_fold(fold_train_df, fold_val_df, folder_output_path, training_config)
            if metric is None:
                raise ValueError("Metric is None!!!")
            eval_trainer_classification_report = f"Fold {fold+1} eval trainer classification report:\n{eval_trainer_classification_report}"
        except Exception as e:
            print(f"Error in fold {fold}: {e}")
            continue 

        # setting best metric and model to best fold until present
        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_fold = fold+1
            best_model = finetuned_fold_model
            best_tokenizer = finetuned_fold_tokenizer

        # storing evaluation classification report for current fold
        reports.append(eval_trainer_classification_report)
    
    if best_fold is not None and best_model is not None:
        print(f"Best fold: {best_fold}/{n_splits}")
        reports[best_fold - 1] = f"BEST FOLD\n{reports[best_fold - 1]}"
        save_metrics(classification_reports=reports, output_path=output_path, filename="reports.txt")

        # storing best model
        model_output_path = os.path.join(output_path, "finetuned_model")
        os.makedirs(model_output_path, exist_ok=True)

        best_model.save_pretrained(model_output_path)
        best_tokenizer.save_pretrained(model_output_path)
        print(f"Best model (from fold {best_fold}) saved to: {model_output_path}")

        # TESTING ==============================================================================
        # creating a test folder for testing results
        test_output_path = os.path.join(output_path, "test")
        os.makedirs(test_output_path)
        print(f"Test folder created at: {test_output_path}")

        # getting model predictions and performances
        predict_df = run_test(test_df=data_config.test_df, 
                                model=best_model, 
                                tokenizer=best_tokenizer, 
                                batch_size=training_config.per_device_eval_batch_size, 
                                output_path=test_output_path)
    
        # storing obtained dataframe into csv file
        predict_path = os.path.join(test_output_path, "predict_df.csv")
        predict_df.to_csv(predict_path)
        print(f"Predict_df stored at: {predict_path}")
        # ========================================================================================

    else:
        print("No valid fold was completed successfully. Cannot evaluate on test set.")
        return None, None, None, None

    return best_fold, best_model, best_tokenizer, predict_df




def main():

    training_config = TrainingConfig()


    baseline_model = CamembertForSequenceClassification.from_pretrained("/home/marina/stageM2/output/clean-baseline/finetuned_model")
    baseline_tokenizer = CamembertTokenizer.from_pretrained("/home/marina/stageM2/output/clean-baseline/finetuned_model")

    clean_stay_clean_baseline_test_path = os.path.join(OUTPUT_DIR, "clean_stay_clean_baseline_test")
    os.makedirs(clean_stay_clean_baseline_test_path, exist_ok=True)

    import pandas as pd
    clean_stay_df = pd.read_csv("/home/marina/stageM2/data/clean_dataset.csv")
    predict_df = run_test(test_df=clean_stay_df,
                            model = baseline_model,
                            tokenizer=baseline_tokenizer,
                            batch_size=training_config.per_device_eval_batch_size,
                            output_path=clean_stay_clean_baseline_test_path)

    # clean_louvain_clean_baseline_test_path = os.path.join(OUTPUT_DIR, "clean_louvain_clean_baseline_test")
    # os.makedirs(clean_louvain_clean_baseline_test_path, exist_ok=True)

    # small_zero_clean_baseline_test_path = os.path.join(OUTPUT_DIR, "small_zeroshot_clean_baseline_test")
    # os.makedirs(small_zero_clean_baseline_test_path, exist_ok=True)

    # import pandas as pd

    # clean_louvain_df = pd.read_csv("/home/marina/stageM2/data/clean_louvain_aug_df.csv")
    # predict_df = run_test(test_df=clean_louvain_df,
    #                      model = baseline_model,
    #                      tokenizer=baseline_tokenizer,
    #                      batch_size=training_config.per_device_eval_batch_size,
    #                      output_path=clean_louvain_clean_baseline_test_path)

    # small_zero_df = pd.read_csv("/home/marina/stageM2/data/zero_shot_pred_df.csv")
    # zero_df = small_zero_df[small_zero_df['text']]
    # predict_df = run_test(test_df=zero_df,
    #                      model = baseline_model,
    #                      tokenizer=baseline_tokenizer,
    #                      batch_size=training_config.per_device_eval_batch_size,
    #                      output_path=small_zero_clean_baseline_test_path)

    # import pandas as pd
    # from sklearn.metrics import classification_report
    # predict_df = pd.read_csv('/home/marina/stageM2/output/clean_louvain_clean_baseline_test/predict_df')

    # ccc = predict_df[predict_df['origin']=='CCC']
    # ccc_report = classification_report(ccc["label"], ccc["pred"], digits=4)
    # save_metrics([ccc_report], output_path="/home/marina/stageM2/output/clean_louvain_clean_baseline_test", filename="ccc_report.txt")

    # cnc = predict_df[predict_df['origin']=='CNC']
    # cnc_report = classification_report(cnc["label"], cnc["pred"], digits=4)
    # save_metrics([cnc_report], output_path="/home/marina/stageM2/output/clean_louvain_clean_baseline_test", filename="cnc_report.txt")

    # ncc = predict_df[predict_df['origin']=='NCC']
    # ncc_report = classification_report(ncc["label"], ncc["pred"], digits=4)
    # save_metrics([ncc_report], output_path="/home/marina/stageM2/output/clean_louvain_clean_baseline_test", filename="ncc_report.txt")

    # nnn = predict_df[predict_df['origin']=='NNN']
    # nnn_report = classification_report(nnn["label"], nnn["pred"], digits=4)
    # save_metrics([nnn_report], output_path="/home/marina/stageM2/output/clean_louvain_clean_baseline_test", filename="nnn_report.txt")
    


if __name__ == "__main__":
    main()