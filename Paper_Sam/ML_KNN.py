import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score
import json
import time # Import time for measuring tuning duration

# Import the specialized MLkNN model
from skmultilearn.adapt import MLkNN

# --- CONFIGURATION ---
PATH_TO_DATA = "./Datasets/"
TRAIN_PATH = PATH_TO_DATA + "train.tsv"
TEST_PATH = PATH_TO_DATA + "test.tsv"
VALIDATION_PATH = PATH_TO_DATA + "dev.tsv"
topic_num_map = {"cs.it": 0, "math.it": 1, "cs.lg": 2, "cs.ai": 3, "stat.ml": 4, "cs.ds": 5, "cs.si": 6, "cs.dm": 7, "physics.soc-ph": 8, "cs.lo": 9, "math.co": 10, "cs.cc": 11, "math.oc": 12, "cs.ni": 13, "cs.cv": 14, "cs.cl": 15, "cs.cr": 16, "cs.sy": 17, "cs.dc": 18, "cs.ne": 19, "cs.ir": 20, "quant-ph": 21, "cs.gt": 22, "cs.cy": 23, "cs.pl": 24, "cs.se": 25, "math.pr": 26, "cs.db": 27, "cs.cg": 28, "cs.na": 29, "cs.hc": 30, "math.na": 31, "cs.ce": 32, "cs.ma": 33, "cs.ro": 34, "cs.fl": 35, "math.st": 36, "stat.th": 37, "cs.dl": 38, "cmp-lg": 39, "cs.mm": 40, "cond-mat.stat-mech": 41, "cs.pf": 42, "math.lo": 43, "stat.ap": 44, "cs.ms": 45, "stat.me": 46, "cs.sc": 47, "cond-mat.dis-nn": 48, "q-bio.nc": 49, "physics.data-an": 50, "nlin.ao": 51, "q-bio.qm": 52, "math.nt": 53}
num_topic_map = {v: k for k, v in topic_num_map.items()}
NUM_LABELS = len(topic_num_map)

class MLkNN_Predictor:
    def __init__(self, k, s):
        # Hyperparameters for MLkNN
        self.k = k
        self.s = s

        # Model and Data Components
        self.vectorizer = None
        self.classifier = None
        self.X_train_raw = None
        self.X_train_sparse = None
        self.Y_train = None

    # --- 1. Data Loading and Preparation ---
    def load_data(self, file_path):
        """Loads data, separates text (X) and converts label string (Y) to matrix."""
        try:
            df = pd.read_csv(file_path, sep="\t", header=None, names=['labels_str', 'text'])
            
            X_raw = df['text'].to_numpy()
            label_lists = [[int(char) for char in label_str] for label_str in df['labels_str']]
            # Ensure Y is a NumPy array
            Y = np.array(label_lists)
            
            return X_raw, Y
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Please check the path.")
            return None, None

    def createTrainDataset(self, train_path):
        """Loads and stores the training data."""
        X_train_raw, Y_train = self.load_data(train_path)
        
        if X_train_raw is not None:
            self.X_train_raw = X_train_raw
            self.Y_train = Y_train
            print(f"Training set loaded with {self.Y_train.shape[0]} samples.")
        
    # --- 2. Feature Extraction ---
    def fit_vectorizer(self):
        """Fits the TF-IDF vectorizer and transforms the training data."""
        if self.X_train_raw is None:
            raise ValueError("Training data must be loaded first! Call createTrainDataset().")

        self.vectorizer = TfidfVectorizer(stop_words='english', min_df=5) 
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train_raw)
        self.X_train_sparse = csr_matrix(X_train_tfidf)
        
        print(f"Features vectorized. X_train sparse shape: {self.X_train_sparse.shape}")

    # --- 3. MLkNN Training ---
    def train_model(self, k=None, s=None):
        """Initializes and trains the MLkNN classifier using current or provided k and s."""
        if self.X_train_sparse is None:
            raise ValueError("Data must be vectorized first! Call fit_vectorizer().")
            
        # Use provided k/s if available, otherwise use instance defaults
        k_val = k if k is not None else self.k
        s_val = s if s is not None else self.s

        self.classifier = MLkNN(k=k_val, s=s_val)
        
        print(f"\nStarting MLkNN Training (k={k_val}, s={s_val})...")
        self.classifier.fit(self.X_train_sparse, self.Y_train)
        print("MLkNN Training Complete.")

    # --- 4. Hyperparameter Tuning Function (New) ---
    def tune_k_and_s_parameters(self, k_values, s_values, validation_path):
        """
        Performs a grid search over k and s values using the validation set.
        """
        X_val_raw, Y_val = self.load_data(validation_path)
        if X_val_raw is None:
            return None, None, []

        X_val_tfidf = self.vectorizer.transform(X_val_raw)
        X_val_sparse = csr_matrix(X_val_tfidf)

        print("\n" + "="*60)
        print(f"🚀 Starting ML-kNN Grid Search Tuning (k: {k_values}, s: {s_values})")
        print("="*60)

        best_k = -1
        best_s = -1
        best_macro_f1 = -1.0
        results_summary = []

        for k in k_values:
            for s in s_values:
                start_time = time.time()
                print(f"\n[k={k}, s={s}] ------------------------------------")
                
                # 1. Train the model with current k and s (re-fitting each time)
                current_classifier = MLkNN(k=k, s=s)
                current_classifier.fit(self.X_train_sparse, self.Y_train)
                
                # 2. Predict on Validation Set
                val_predictions_sparse = current_classifier.predict(X_val_sparse)
                val_predictions = val_predictions_sparse.toarray()

                # 3. Calculate Macro F1-Score (Primary tuning metric)
                macro_f1 = f1_score(Y_val, val_predictions, average='macro', zero_division=0)
                h_loss = hamming_loss(Y_val, val_predictions)
                
                elapsed_time = time.time() - start_time
                
                print(f"  -> Macro F1-Score: {macro_f1:.4f}")
                print(f"  -> Hamming Loss: {h_loss:.4f}")
                print(f"  -> Time taken: {elapsed_time:.2f}s")

                results_summary.append({
                    "k": k,
                    "s": s,
                    "Macro_F1": macro_f1,
                    "Hamming_Loss": h_loss,
                    "Time_Seconds": elapsed_time
                })

                # 4. Check if this is the best combination
                if macro_f1 > best_macro_f1:
                    best_macro_f1 = macro_f1
                    best_k = k
                    best_s = s

        print("\n" + "="*60)
        print(f"✅ Fine-Tuning Complete. Best (k, s) found: **({best_k}, {best_s})** (Macro F1: {best_macro_f1:.4f})")
        print("="*60)
        
        # Update the instance's best hyperparameters
        self.k = best_k
        self.s = best_s
        
        return best_k, best_s, results_summary
        
    # --- 5. Prediction and Evaluation (Modified to take raw data directly) ---
    def predict_and_evaluate(self, X_test_raw, Y_test, outPath):
        """Uses the trained model to predict, report metrics, and save results."""
        
        if self.classifier is None:
            print("Cannot run prediction: Trained model missing. Call train_model() first.")
            return

        # Transform test data using the FITTED vectorizer
        X_test_tfidf = self.vectorizer.transform(X_test_raw)
        X_test_sparse = csr_matrix(X_test_tfidf)
        
        # Make predictions
        mlknn_predictions_sparse = self.classifier.predict(X_test_sparse)
        mlknn_predictions = mlknn_predictions_sparse.toarray()
        
        # 1. Calculate Comprehensive Metrics
        s_accuracy = accuracy_score(Y_test, mlknn_predictions) 
        h_loss = hamming_loss(Y_test, mlknn_predictions)
        j_score_macro = jaccard_score(Y_test, mlknn_predictions, average='macro', zero_division=0)
        f1_macro = f1_score(Y_test, mlknn_predictions, average='macro', zero_division=0)
        f1_micro = f1_score(Y_test, mlknn_predictions, average='micro', zero_division=0)

        print("\n--- MLkNN Performance on Test Set (Multi-Label Metrics) ---")
        print(f"Test Samples: {Y_test.shape[0]}")
        print(f"Hyperparameters: k={self.k}, s={self.s}")
        print("-" * 50)
        print(f"1. **Subset Accuracy (Exact Match)**: {s_accuracy:.4f}")
        print(f"2. **Hamming Loss**: {h_loss:.4f}")
        print(f"3. **Macro F1-Score**: {f1_macro:.4f}")
        print(f"4. **Micro F1-Score**: {f1_micro:.4f}")
        print("-" * 50)

        # 2. Save Results to JSON 
        results_list = []
        
        for i in range(mlknn_predictions.shape[0]):
            predicted_topic_indices = np.where(mlknn_predictions[i] == 1)[0]
            predicted_topic_names = [num_topic_map[idx] for idx in predicted_topic_indices]
            
            true_topic_indices = np.where(Y_test[i] == 1)[0]
            true_topic_names = [num_topic_map[idx] for idx in true_topic_indices]
            
            results_list.append({
                "index": i,
                "prediction": predicted_topic_names, 
                "true_labels": true_topic_names, 
                "test_document_text_start": X_test_raw[i][:100] + "..." 
            })
            
        # Include metrics in the final output file
        output_data = {
            "model_name": "MLkNN",
            "hyperparameters": {"k": self.k, "s": self.s},
            "metrics": {
                "subset_accuracy": s_accuracy,
                "hamming_loss": h_loss,
                "jaccard_score_macro": j_score_macro,
                "f1_score_macro": f1_macro,
                "f1_score_micro": f1_micro
            },
            "predictions": results_list
        }

        with open(outPath, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        print(f"\n✅ Results saved successfully to {outPath}")
        
        # Example of one prediction with topic names
        if results_list:
            first_result = results_list[0]
            print("\nExample Prediction (First Sample with Topic Names):")
            print(f"True Labels: {first_result['true_labels']}")
            print(f"Predicted Labels: {first_result['prediction']}")


# --- EXECUTION ---

if __name__ == "__main__":
    
    # 1. Initialize the predictor (Initial k/s values are placeholders for tuning)
    model = MLkNN_Predictor(k=10, s=1.0) 
    
    # 2. Load all data
    trainRaw, Y_train = model.load_data(TRAIN_PATH)
    valRaw, Y_val = model.load_data(VALIDATION_PATH)
    testRaw, Y_test = model.load_data(TEST_PATH)
    
    # Load and store training data (to populate X_train_raw and Y_train)
    if trainRaw is not None:
        model.X_train_raw = trainRaw
        model.Y_train = Y_train
        print(f"Training set loaded with {model.Y_train.shape[0]} samples.")
    
    # 3. Vectorize Features (Must be done once on training data)
    model.fit_vectorizer()
    
    # 4. Define k and s ranges for tuning
    k_range = [5, 10, 15, 20] # Number of neighbors
    s_range = [0.1, 1.0, 5.0]  # Smoothing parameter
    
    # 5. Fine-tune k and s using the validation set
    best_k, best_s, tuning_results = model.tune_k_and_s_parameters(
        k_values=k_range, 
        s_values=s_range, 
        validation_path=VALIDATION_PATH
    )
    
    # Optional: Save tuning results to a file
    with open("mlknn_tuning_results.json", 'w') as f:
        json.dump(tuning_results, f, indent=4)
    print(f"\nTuning summary saved to mlknn_tuning_results.json")
    
    # 6. Train the final model using the best parameters
    # The instance's k and s were updated in tune_k_and_s_parameters
    model.train_model() 
    
    # 7. Predict and Evaluate on Test Set
    model.predict_and_evaluate(testRaw, Y_test, f"../output/dev_ML-kNN_k{best_k}_s{best_s}.json")