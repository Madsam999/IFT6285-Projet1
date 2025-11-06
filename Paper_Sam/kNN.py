import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score
import json

PATH_TO_DATA = "./Datasets/"
PATH_TO_TRAIN = "train.tsv"
PATH_TO_VAL = "dev.tsv"
PATH_TO_TEST = "test.tsv"

class kNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.Y_train_i = None

    def getNeighbor(self, point):
        distances = euclidean_distances(point, self.X_train)
        distances = distances.flatten()
        sortedDist = np.argsort(distances)
        return sortedDist[:self.k]
    
    def vote(self, neighbors):
        neighbor_labels = self.Y_train_i[neighbors]
        positiveVote = np.sum(neighbor_labels)

        if positiveVote > self.k / 2.0:
            return 1
        else:
            return 0
        
    def fit(self, X_train, Y_train_i, fitted_vectorizer = None):
        print(f"Fitting binary classifier kNN")
        if fitted_vectorizer == None:
            self.vectorizer = TfidfVectorizer(stop_words="english", min_df = 5)
        else:
            self.vectorizer = fitted_vectorizer
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.X_train = csr_matrix(X_train_tfidf)
        self.Y_train_i = Y_train_i

    def predict(self, text):
        X_test = self.vectorizer.transform([text])
        neighbors = self.getNeighbor(X_test)
        return self.vote(neighbors)

class BinaryRelevanceKNN: 
    def __init__(self, k, numLabels):
        self.numLabels = numLabels
        self.classifiers = [kNNClassifier(k) for _ in range(numLabels)]
        self.k = k
        self.fitted_vectorizer = None

    def fit(self, X_train, Y_train):
        print(f"Fitting the Binary relevance kNN")
        Y_train_array = np.array(Y_train)

        self.classifiers[0].fit(X_train, Y_train[:, 0])
        self.fitted_vectorizer = self.classifiers[0].vectorizer

        for i in range(self.numLabels):
            Y_train_i = Y_train_array[:, i]
            if i == 0:
                continue
            else:
                self.classifiers[i].fit(X_train, Y_train_i, fitted_vectorizer = self.fitted_vectorizer)

    def predict(self, X_test):
        print(f"Predidicting set of labels")
        predictions = []
        for text in X_test:
            predictions_i = []
            for i in range(self.numLabels):
                predictions_i.append(self.classifiers[i].predict(text))
            predictions.append(predictions_i)
        return np.array(predictions)

def processData(pathToData):
    dataFrame = pd.read_csv(pathToData, sep = "\t", header = None, names = ["labels_str", "text"])
    xRaw = dataFrame["text"]
    labelsList = [[int(char) for char in label_str] for label_str in dataFrame["labels_str"]]
    Y = np.array(labelsList)
    return xRaw, Y

trainRaw, Y_train = processData(PATH_TO_DATA + PATH_TO_TRAIN)
valRaw, Y_val = processData(PATH_TO_DATA + PATH_TO_VAL)
testRaw, Y_test = processData(PATH_TO_DATA + PATH_TO_TEST)

model = BinaryRelevanceKNN(10, 54)

model.fit(trainRaw, Y_train)

predictions = model.predict(testRaw)

print(f"\nPredictions shape: {predictions.shape}")

# --- METRICS CALCULATION (ADDED) ---
h_loss = hamming_loss(Y_test, predictions)
s_accuracy = accuracy_score(Y_test, predictions) # Subset Accuracy / Exact Match
f1_macro = f1_score(Y_test, predictions, average='macro', zero_division=0)
f1_micro = f1_score(Y_test, predictions, average='micro', zero_division=0)
# ------------------------------------

print("-" * 40)
print("Binary Relevance kNN Evaluation:")
print("-" * 40)
print(f"1. **Subset Accuracy (Exact Match)**: {s_accuracy:.4f}")
print(f"2. **Macro F1-Score**: {f1_macro:.4f}")
print(f"3. **Micro F1-Score**: {f1_micro:.4f}")
print(f"4. **Hamming Loss**: {h_loss:.4f}")
print("-" * 40)