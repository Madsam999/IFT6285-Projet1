import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score
#from topic_num_map import topic_num_map
import json

# Import the specialized MLkNN model
# NOTE: Ensure 'scikit-multilearn' is installed and compatible with your scikit-learn version.
from skmultilearn.adapt import MLkNN

# --- CONFIGURATION ---
PATH_TO_DATA = "./Datasets/"
TRAIN_PATH = PATH_TO_DATA + "train.tsv"
TEST_PATH = PATH_TO_DATA + "test.tsv"
VALIDATION_PATH = PATH_TO_DATA + "dev.tsv"
topic_num_map = {"cs.it": 0, "math.it": 1, "cs.lg": 2, "cs.ai": 3, "stat.ml": 4, "cs.ds": 5, "cs.si": 6, "cs.dm": 7, "physics.soc-ph": 8, "cs.lo": 9, "math.co": 10, "cs.cc": 11, "math.oc": 12, "cs.ni": 13, "cs.cv": 14, "cs.cl": 15, "cs.cr": 16, "cs.sy": 17, "cs.dc": 18, "cs.ne": 19, "cs.ir": 20, "quant-ph": 21, "cs.gt": 22, "cs.cy": 23, "cs.pl": 24, "cs.se": 25, "math.pr": 26, "cs.db": 27, "cs.cg": 28, "cs.na": 29, "cs.hc": 30, "math.na": 31, "cs.ce": 32, "cs.ma": 33, "cs.ro": 34, "cs.fl": 35, "math.st": 36, "stat.th": 37, "cs.dl": 38, "cmp-lg": 39, "cs.mm": 40, "cond-mat.stat-mech": 41, "cs.pf": 42, "math.lo": 43, "stat.ap": 44, "cs.ms": 45, "stat.me": 46, "cs.sc": 47, "cond-mat.dis-nn": 48, "q-bio.nc": 49, "physics.data-an": 50, "nlin.ao": 51, "q-bio.qm": 52, "math.nt": 53}
num_topic_map = {v: k for k, v in topic_num_map.items()}
class MLkNN_Predictor:
    def __init__(self, k, s):
        # Hyperparameters for MLkNN
        self.k = k
        self.s = s

        # Model and Data Components
        self.vectorizer = None
        self.classifier = None
        self.X_train_sparse = None
        self.Y_train = None

    # --- 1. Data Loading and Preparation (Modified to be outside the class) ---
    def load_data(self, file_path):
        """Loads data, separates text (X) and converts label string (Y) to matrix."""
        try:
            df = pd.read_csv(file_path, sep="\t", header=None, names=['labels_str', 'text'])
            
            X_raw = df['text'].to_numpy()
            label_lists = [[int(char) for char in label_str] for label_str in df['labels_str']]
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

    # --- 3. MLkNN Training (Using the Sklearn-Multilearn Example) ---
    def train_model(self):
        """Initializes and trains the MLkNN classifier using the library's fit method."""
        if self.X_train_sparse is None:
            raise ValueError("Data must be vectorized first! Call fit_vectorizer().")

        # Initialize the MLkNN classifier using the library's approach
        self.classifier = MLkNN(k=self.k, s=self.s)
        
        print(f"\nStarting MLkNN Training (k={self.k}, s={self.s})...")
        # --- THE KEY LINE FROM THE EXAMPLE PAGE ---
        # This line internally performs the kNN search for all training examples (prior calculation).
        self.classifier.fit(self.X_train_sparse, self.Y_train)
        # ----------------------------------------
        
        print("MLkNN Training Complete.")

    # --- 4. Prediction and Evaluation ---
    def predict_and_evaluate(self, test_path, outPath):
        """Loads test data, predicts, and reports metrics."""
        
        X_test_raw, Y_test = self.load_data(test_path)
        
        if X_test_raw is None or self.classifier is None:
            print("Cannot run prediction: Test data or trained model missing.")
            return

        # Transform test data using the FITTED vectorizer
        X_test_tfidf = self.vectorizer.transform(X_test_raw)
        X_test_sparse = csr_matrix(X_test_tfidf)
        
        # Make predictions
        mlknn_predictions_sparse = self.classifier.predict(X_test_sparse)
        mlknn_predictions = mlknn_predictions_sparse.toarray()
        
        # 1. Calculate Comprehensive Metrics (No change here, metrics are robust)
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

        # 2. Save Results to JSON (MODIFIED to use topic names)
        
        # Create the list of predictions using the specified format
        results = []
        
        for i in range(mlknn_predictions.shape[0]):
            
            # --- CONVERT INDICES TO HUMAN-READABLE TOPIC NAMES ---
            # Get the indices of the predicted positive labels (where prediction == 1)
            predicted_topic_indices = np.where(mlknn_predictions[i] == 1)[0]
            # Map indices (0, 1, 2...) to names ("cs.it", "math.it", "cs.lg"...)
            predicted_topic_names = [num_topic_map[idx] for idx in predicted_topic_indices]
            
            # Get the indices of the true positive labels
            true_topic_indices = np.where(Y_test[i] == 1)[0]
            true_topic_names = [num_topic_map[idx] for idx in true_topic_indices]
            # ----------------------------------------------------
            
            results.append({
                "index": i,
                "prediction": predicted_topic_names, # List of predicted topic names
                "true_labels": true_topic_names,      # List of true topic names
                "test_document_text_start": X_test_raw[i][:100] + "..." # Optional: first 100 chars for context
            })

        with open(outPath, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✅ Results saved successfully to {outPath}")
        
        # Example of one prediction with topic names
        if results:
            first_result = results[0]
            print("\nExample Prediction (First Sample with Topic Names):")
            print(f"True Labels: {first_result['true_labels']}")
            print(f"Predicted Labels: {first_result['prediction']}")


# --- EXECUTION ---

if __name__ == "__main__":
    
    # 1. Initialize the predictor
    model = MLkNN_Predictor(25, 5) 
    
    # 2. Load Training Data
    model.createTrainDataset(TRAIN_PATH)
    
    # 3. Vectorize Features
    model.fit_vectorizer()
    
    # 4. Train the MLkNN Model (Uses the fit method from the example)
    model.train_model()
    
    # 5. Predict and Evaluate
    model.predict_and_evaluate(TEST_PATH, "../output/dev")