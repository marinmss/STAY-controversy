from dataclasses import dataclass, asdict, field
import json
import os
import pandas as pd
from typing import Optional


# DATA CONFIGURATION ==============================================================================
@dataclass
class DatasetConfig:
    train_version: str = "original-dataset"
    test_version: str = "original-dataset"
    train_path: str = "/home/marina/stageM2/data/dataset.csv"
    test_path: str = "/home/marina/stageM2/data/dataset.csv"
    train_df = None
    test_df = None
    text_column: str = "text"
    label_column: str = "label"

    train_df: Optional[pd.DataFrame] = field(default=None, init=False)
    test_df: Optional[pd.DataFrame] = field(default=None, init=False)

    def __post_init__(self):
        self.load()

    def load(self):
        """
        Load train and test pandas dataframes into corresponding attributes.
        """
        # checking if train_df parameters is valid
        if self.train_df is None:
            self.train_df = pd.read_csv(self.train_path)

        # checking if train_df parameters is valid
        if self.test_df is None:
            self.test_df = pd.read_csv(self.test_path)

        # checking if required columns present in both the training and the evaluation dataframes
        for name,df in [("train",self.train_df), ("test",self.test_df)]:
            for column, label in [(self.text_column, "text"),(self.label_column, "label")]:
                if column not in df.columns:
                    print(f"No {label} column in {name} dataframe")


    def save(self, output_path: str, filename:str = "data_config.json"):
        """
        Save data configuration into a json file.
        """

        # defining saving filepath
        filepath = os.path.join(output_path, filename)

        # storing dataset configuration into json file
        with open(filepath, "w") as f:
            config_dict = {k: v for k, v in asdict(self).items() 
                          if not k.endswith('_df')}
            json.dump(config_dict, f, indent=4)







# TRAINING CONFIGURATION ======================================================================
@dataclass
class TrainingConfig:
    learning_rate : float = 2e-5
    warmup_steps : int = 500
    per_device_train_batch_size : int = 16
    per_device_eval_batch_size : int = 16
    num_train_epochs : int = 10
    evaluation_strategy : str = "epoch"
    save_strategy : str = "epoch"
    logging_strategy="steps"
    logging_dir : str = ".logs"
    logging_steps : int = 10
    load_best_model_at_end : bool = True
    metric_for_best_model : str = "accuracy"
    save_total_limit = 1
    output_dir : str = ".output"
    report_to=["wandb"]



    def save(self, output_path:str, filename:str = "train_config.json"):
        """
        Save Trainer training parameters configuration into a json file.
        """

        # defining saving filepath
        filepath = os.path.join(output_path, filename)

        # storing training configuration into json file
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)






# CROSS VALIDATION CONFIGURATION =============================================================
@dataclass
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42



    def save(self, output_path:str, filename:str = "cval_config.json"):
        """
        Save cross validation configuration into a json file.
        """

        # defining saving filepath
        filepath = os.path.join(output_path, filename)

        # storing cross validation configuration into json file
        with open(filepath, "w") as f:
            json.dump(asdict(self), f, indent=4)
