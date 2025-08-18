# Study of controversy in the agricultural field using French-language comments from YouTube
This code results from the research work conducted during a 6-month internship was done at INRAE, UMR TETIS in Montpellier from March 3rd to September 2nd 2025.
It is organized in 7 folders:

### data
Contains the initial STAY comments in the json format, and the pandas dataframes used in the rest of the internship.

### statistical_analysis
Contains all files linked to the statistical analysis of the STAY corpus to find relevant controversy markers in the data.

### finetuning_pipeline
Contains all files linked to the establishement of a finetuning pipeline to finetune CamemBERT for controversy classification.

### augmentation_methods
Contains all files which allow for the augmentation of the STAY corpus with the two methods explored, cross-over and prompting.

### keywords
Contains a keywords evaluation pipeline for thematic keywords linked to the topic labels. This code was not retained in the project but could be used
in possible continuations.

### **Tips**
The main.py script and the main() function it contains allow for the running of the statistical analysis as well as the running of the cross-over augmentation method.
Make sure to adapt the data and output paths.

For all scripts apart from the finetuning pipeline, the utilized pip virtual environment can be extracted using the following command line:
pip install -r requirements.txt
