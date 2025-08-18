from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class Config:
    def __init__(self, df, embedding_model, isolation_model):
        self.df = df
        self.corpus = list(df['text'])
        self.embedding_model = embedding_model
        self.isolation_model = isolation_model
        
        self.corpus_embeddings = self.embedding_model.encode(self.corpus, show_progress_bar=True)