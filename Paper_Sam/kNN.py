import pandas as pd
import numpy as np
import time # Added for timing the tuning process
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score
import json

# Define the topic mapping for human-readable output (Assuming it was available in the original context)
topic_num_map = {"cs.it": 0, "math.it": 1, "cs.lg": 2, "cs.ai": 3, "stat.ml": 4, "cs.ds": 5, "cs.si": 6, "cs.dm": 7, "physics.soc-ph": 8, "cs.lo": 9, "math.co": 10, "cs.cc": 11, "math.oc": 12, "cs.ni": 13, "cs.cv": 14, "cs.cl": 15, "cs.cr": 16, "cs.sy": 17, "cs.dc": 18, "cs.ne": 19, "cs.ir": 20, "quant-ph": 21, "cs.gt": 22, "cs.cy": 23, "cs.pl": 24, "cs.se": 25, "math.pr": 26, "cs.db": 27, "cs.cg": 28, "cs.na": 29, "cs.hc": 30, "math.na": 31, "cs.ce": 32, "cs.ma": 33, "cs.ro": 34, "cs.fl": 35, "math.st": 36, "stat.th": 37, "cs.dl": 38, "cmp-lg": 39, "cs.mm": 40, "cond-mat.stat-mech": 41, "cs.pf": 42, "math.lo": 43, "stat.ap": 44, "cs.ms": 45, "stat.me": 46, "cs.sc": 47, "cond-mat.dis-nn": 48, "q-bio.nc": 49, "physics.data-an": 50, "nlin.ao": 51, "q-bio.qm": 52, "math.nt": 53}
num_topic_map = {v: k for k, v in topic_num_map.items()}
NUM_LABELS = len(topic_num_map)

PATH_TO_DATA = "./Datasets/"
PATH_TO_TRAIN = "train.tsv"
PATH_TO_VAL = "dev.tsv"
PATH_TO_TEST = "test.tsv"

class kNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.Y_train_i = None
        self.vectorizer = None # Ensure vectorizer is initialized

    def getNeighbor(self, point):
        distances = euclidean_distances(point, self.X_train)
        distances = distances.flatten()
        sortedDist = np.argsort(distances)
        return sortedDist[:self.k]
    
    def vote(self, neighbors):
        neighbor_labels = self.Y_train_i[neighbors]
        positiveVote = np.sum(neighbor_labels)

        # Standard kNN majority vote logic
        if positiveVote > self.k / 2.0:
            return 1
        else:
            return 0
        
    def fit(self, X_train, Y_train_i, fitted_vectorizer = None):
        if fitted_vectorizer is None:
            self.vectorizer = TfidfVectorizer(stop_words="english", min_df = 5)
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
        else:
            self.vectorizer = fitted_vectorizer
            X_train_tfidf = self.vectorizer.transform(X_train)
            
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
        print(f"Fitting the Binary relevance kNN with initial k={self.k}")
        Y_train_array = np.array(Y_train)

        # Fit the first classifier and get the fitted vectorizer
        self.classifiers[0].fit(X_train, Y_train[:, 0])
        self.fitted_vectorizer = self.classifiers[0].vectorizer

        # Fit the remaining classifiers using the *same* vectorizer
        for i in range(1, self.numLabels):
            Y_train_i = Y_train_array[:, i]
            # Print progress every 10 labels
            if i % 10 == 0:
                 print(f"    - Fitting label {i}/{self.numLabels}...")
            self.classifiers[i].fit(X_train, Y_train_i, fitted_vectorizer = self.fitted_vectorizer)
        print("Binary Relevance kNN Training Complete.")


    def predict(self, X_test):
        # Optimized: If we have many samples, we can pre-vectorize the test set
        # This implementation requires predicting one text at a time due to the inner kNN,
        # so we keep the loop structure but ensure the vectorizer is ready.
        predictions = []
        for text in X_test:
            predictions_i = []
            for classifier in self.classifiers:
                predictions_i.append(classifier.predict(text))
            predictions.append(predictions_i)
        return np.array(predictions)

# -----------------------------------------------------
## 1. K-Parameter Fine-Tuning Function 🔍
# -----------------------------------------------------
def tune_k_parameter(k_values, trainRaw, Y_train, valRaw, Y_val):
    """
    Evaluates the BR-kNN model across a range of k values using the validation set.

    Args:
        k_values (list): A list of integer k values to test.
        trainRaw, Y_train: Training data (text and labels).
        valRaw, Y_val: Validation data (text and labels).

    Returns:
        tuple: (best_k, best_macro_f1, results_summary)
    """
    print("\n" + "="*60)
    print("🚀 Starting K-Parameter Fine-Tuning on Validation Set...")
    print(f"Testing k values: {k_values}")
    print("="*60)

    best_k = -1
    best_macro_f1 = -1.0
    results_summary = []

    for k in k_values:
        start_time = time.time()
        print(f"\n[K={k}] ------------------------------------")
        
        # 1. Initialize and Fit Model for the current k
        brknn_model = BinaryRelevanceKNN(k=k, numLabels=NUM_LABELS)
        brknn_model.fit(trainRaw, Y_train)
        
        # 2. Predict on Validation Set
        val_predictions = brknn_model.predict(valRaw)

        # 3. Calculate Macro F1-Score
        macro_f1 = f1_score(Y_val, val_predictions, average='macro', zero_division=0)
        
        # Calculate other metrics for logging
        h_loss = hamming_loss(Y_val, val_predictions)
        s_accuracy = accuracy_score(Y_val, val_predictions)

        elapsed_time = time.time() - start_time
        
        print(f"  -> Macro F1-Score: {macro_f1:.4f}")
        print(f"  -> Hamming Loss: {h_loss:.4f}")
        print(f"  -> Time taken: {elapsed_time:.2f}s")

        results_summary.append({
            "k": k,
            "Macro_F1": macro_f1,
            "Subset_Accuracy": s_accuracy,
            "Hamming_Loss": h_loss,
            "Time_Seconds": elapsed_time
        })

        # 4. Check if this is the best k
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_k = k

    print("\n" + "="*60)
    print(f"✅ Fine-Tuning Complete. Best k found: **{best_k}** (Macro F1: {best_macro_f1:.4f})")
    print("="*60)
    
    return best_k, best_macro_f1, results_summary

# -----------------------------------------------------
## 2. Prediction and Evaluation Function (Unchanged)
# -----------------------------------------------------
def predict_and_evaluate(model, X_test_raw, Y_test, output_filename="dev_BR-kNN.json"):
    """
    Performs prediction using the BRkNN model, calculates metrics, 
    and saves detailed results and metrics to a JSON file.
    """
    print(f"\n--- Starting Prediction for BRkNN (k={model.k}) on Test Set ---")
    predictions = model.predict(X_test_raw)
    
    print(f"Predictions shape: {predictions.shape}")

    # 1. Calculate Comprehensive Metrics
    h_loss = hamming_loss(Y_test, predictions)
    s_accuracy = accuracy_score(Y_test, predictions) # Subset Accuracy / Exact Match
    j_score_macro = jaccard_score(Y_test, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(Y_test, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(Y_test, predictions, average='micro', zero_division=0)

    # 2. Print Summary Metrics
    print("\n--- Binary Relevance kNN Evaluation Summary ---")
    print("-" * 50)
    print(f"Model Hyperparameter: k={model.k}")
    print(f"1. **Subset Accuracy (Exact Match)**: {s_accuracy:.4f}")
    print(f"2. **Hamming Loss**: {h_loss:.4f}")
    print(f"3. **Macro F1-Score**: {f1_macro:.4f}")
    print(f"4. **Micro F1-Score**: {f1_micro:.4f}")
    print("-" * 50)

    # 3. Prepare detailed results and metrics for JSON output
    results_list = []
    
    for i in range(predictions.shape[0]):
        # Convert indices to human-readable topic names
        predicted_topic_indices = np.where(predictions[i] == 1)[0]
        true_topic_indices = np.where(Y_test[i] == 1)[0]
        
        predicted_topic_names = [num_topic_map[idx] for idx in predicted_topic_indices if idx in num_topic_map]
        true_topic_names = [num_topic_map[idx] for idx in true_topic_indices if idx in num_topic_map]
        
        results_list.append({
            "index": i,
            "prediction": predicted_topic_names, 
            "true_labels": true_topic_names, 
            "test_document_text_start": X_test_raw[i][:100] + "..." 
        })
    
    output_data = {
        "model_name": "Binary Relevance kNN",
        "hyperparameter_k": model.k,
        "metrics": {
            "subset_accuracy": s_accuracy,
            "hamming_loss": h_loss,
            "jaccard_score_macro": j_score_macro,
            "f1_score_macro": f1_macro,
            "f1_score_micro": f1_micro
        },
        "predictions": results_list
    }

    # 4. Save results to JSON file
    try:
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"\n✅ Results successfully saved to **{output_filename}**")
    except Exception as e:
        print(f"\n❌ Error saving JSON file: {e}")

    return predictions


def processData(pathToData):
    dataFrame = pd.read_csv(pathToData, sep = "\t", header = None, names = ["labels_str", "text"])
    xRaw = dataFrame["text"]
    labelsList = [[int(char) for char in label_str] for label_str in dataFrame["labels_str"]]
    # Pad the label lists if necessary, although the data loading seems to imply a fixed-length string of 0s/1s
    Y = np.array(labelsList)
    return xRaw, Y

# --- EXECUTION BLOCK (Modified to include fine-tuning) ---

if __name__ == "__main__":
    
    # 1. Load Data
    trainRaw, Y_train = processData(PATH_TO_DATA + PATH_TO_TRAIN)
    valRaw, Y_val = processData(PATH_TO_DATA + PATH_TO_VAL)
    testRaw, Y_test = processData(PATH_TO_DATA + PATH_TO_TEST)

    # 2. Define the range of k values to test
    # A common starting point is a small set of odd numbers to break ties.
    k_range = [3, 5, 7, 11, 15, 21, 25] 
    
    
    # 4. Train the final model using the best k
    final_model = BinaryRelevanceKNN(k=10, numLabels=NUM_LABELS)
    final_model.fit(trainRaw, Y_train)
    
    # 5. Predict and Evaluate on Test Set
    predict_and_evaluate(
        final_model, 
        testRaw, 
        Y_test, 
        output_filename=f"dev_BR-kNN_k{10}.json"
    )