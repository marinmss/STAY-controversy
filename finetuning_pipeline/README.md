# Finetuning pipeline
A pipeline to finetune CamembertForSequenceClassification using Trainer from Transformers.
Contains the initial finetuning pipeline in the initial_pipeline folder.
The pipeline with cross-validation implemented is contained in the cross_validation folder. The scripts are detailed below

### configurations.py
Defines the classes for the configuration objects:
- DatasetConfig: defines the training and testing pandas DataFrames
- TrainingConfig: defines the TrainingArguments for the instanciation of the Trainer object, i.e the hyperparameters for the finetuning process
- CVConfig: defines the cross-validation configuration such as the number of splits and the random seed

### logger.py
Contains all functions which serves to obtain a comparable output through the generation of textual files.

### plotter.py
Contains all functions which serve to obtain a comparable output through the generation of plots and figures.

### tester.py
Contains all function which allow for the testing of a finetuned model on an external dataset.

# finetuner.py
Serves as an entry point function to the pipeline. The main function is utilized to define the DatasetConfig, TrainingConfig and CVConfig objects and initiate the training.
The training is called by the finetune_cv() function, to which the configuration objects are to be passed as parameters


### **Tips**
Call the finetune_cv() function from finetuner.py script to initiate the training.
The running of these scripts require the use of a GPU.
Make sure to adapt the data and output paths.

A conda environment was utilized for this particular implementation.    
The requirements are contained in the finetuning_environment.yml file.
The corresponding virtual environement can be extracted using the following command line:
conda env create -f finetuning_environment.yml