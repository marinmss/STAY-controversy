# Statistical analysis
A comparative study of textual markers in the 1400 STAY comments. Includes 2 control corpora, the blog corpus (5 pages) and the wikipedia corpus (14 pages).

### blog_corpus.py
Builds the blog corpus. Stores the 5 blog pages into a pandas DataFrame with a 'text' column. Each page is treated as a textual unit and stored in a different row.

### wiki_corpus.py
Builds the Wikipedia corpus. Stores the 14 Wikipedia pages into a pandas DataFrame with a 'text' column. Each page is treated as a textual unit and stored in a different row.

### analysis_df.py
The function analyze_df allows for the computation of the following features per textual unit, given a pandas DataFrame:
- length (number of characters)
- number of words
- number of punctuation marks
- number of pos pos_tag occurences
- number of emojis
- number of URLs
- number of stopwords
- the sentiment (polarity and subjectivity) scores
- the number of named entities
Returns a pandas DataFrame with a feature per column, and a pandas DataFrame with normalized features.

### freq.py
Allows for the retrieval of the most frequent ngrams and most frequent occurencies of a given POS value in a corpus.

### stats.py
Computes the mean, standard deviation and quartiles for each feature given a pandas DataFrame.

### plot_utils.py
Allows for the visual analysis of the features through the plotting of radar plots, boxplots, barplots and piecharts.

### run_statistical_analysis.py
Defines the corpora and stores them into separate DataFrames
- controversial comments corpus
- non-controversial comments corpus
- blog corpus
- wikipedia corpus
Serves as an entry point function to perform the statistical analysis.


### **Tips**
Run the run_statistical_analysis.py script.
Make sure to adapt the data and output paths.