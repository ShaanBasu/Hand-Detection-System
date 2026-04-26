"""
Gesture Classifier Script: 
This script  handles training and prediction of hand gestures using machine learning
"""

import numpy as np
import pickle
import os
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


"""
This following class is the machine learning model for gesture classification
It loads training data from a pickle files, trains a  random forest classsifier, saves/loads trained models and then makes predicitons based on new hand landmarks
"""

class GestureClassifier:
    
    """
    This function initialises the gesture classifier class
    """
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == "random_forest":
            self.clf = RandomForestClassifier(
                n_estimators=100,        # Number of trees in forest
                max_depth=15,            # Max depth of each tree
                min_samples_split=5,     # Min samples to split node
                min_samples_leaf=2,      # Min samples at leaf
                random_state=42,         # For reproducibility
                n_jobs=-1                # Use all CPU cores
            )
            
        else: 
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_trained = False
        self.gesture_labels = None
        
    
    """
    This function loads all the training data from the pickle files
    It returns a tuple of (X,y) where  X is the numpy array of  shape (n_samples, 63) - landmark data and y is the numpy array of (n_samples,) - gesture labels
    """
    def load_training_data(self, data_dir: str = "data/instrument_gestures") -> Tuple[np.ndarray, np.ndarray]:
        X = [] #Feature data / landmarks
        y = [] #Labels / gesture names
        
        #First we descend into subfolder containing the data for the instruments
        #os.walk descends into the subfolder(root being the folder path being visited currently, dirs being the list of subfolders inside root, files being the name of files inside root)
        for root, dirs, files in os.walk(data_dir):
            for filename in files:
                if not filename.endswith(".pkl"):
                    continue # This skips any non pickle files
                
                #The gesture name of the instrument will be the filename without the .pkl extension
                # for example piano_ready.pkl -> "piano_ready"
                gesture_name = filename.replace(".pkl", "")
                file_path = os.path.join(root,filename)
                
                
                try:
                    with open(file_path, "rb") as f:
                        gesture_landmarks = pickle.load(f)
                        # The gesture landmarks will be a list of samples.
                        #Each sample is a list of 63 floats(21 landmarks with 3 x,y,z coordinates)
                        for landmarks in gesture_landmarks:
                            X.append(landmarks)
                            y.append(gesture_name)
                    
                    print(f"Successfully loaded {gesture_name}: {len(gesture_landmarks)} samples")
                    
                except Exception as e:
                    print(f" Error loading {file_path}: {e}")
            
        if not X:
            raise FileNotFoundError(f"No gesture data found under '{data_dir}'")
        
        #We then convert python lists to numpy arrays for use with scikit learn
        X = np.array(X)
        y = np.array(y)
        
        #We then store the unique gesture names so predict() method can reference them later
        self.gesture_labels = np.unique(y)
        
        print(f"\nTotal samples loaded: {len(X)}")
        print(f"Gesture labels found: {list(self.gesture_labels)}")
        
        return X, y
    
    
    """
    This function trains the gesture classifier 
    test_size: Is the fraction of data to use for testing
     Args:
            X:         Feature matrix, shape (n_samples, 63)
            y:         Label array, shape (n_samples,)
            test_size: Fraction of data to hold out for testing (default 20%)

        Returns:
            test_accuracy: float — the accuracy on the held-out test set
    """
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        
        print("\n" + "=" * 60)
        print("Training Gesture Classifier")
        print("=" *60)
        
        
        # Step 1
        # Split the data into training and testing sets
        # stratify=y ensures each gesture class is equally represented
        # in both splits, preventing a biased evaluation.
        print(f"\nSplitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples    : {len(X_test)}")
        
        #Step 2: We normalise the features
        # StandardScaler centres each feature to mean=0 and scales to std=1.
        # We fit ONLY on training data (so test data stays unseen during fitting),
        # then transform both splits using the same fitted scaler.
        print("\nNormalising Features... ")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaled.transform(X_test)
        
        #Step 3: We train the random forest
        print("Fitting Random ForesT Classifier...")
        self.clf.fit(X_train_scaled, y_train)
        
        #Step 4: We evaluate the trained random forest on the test set
        y_pred = self.clf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'=' * 60}")
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        print(f"{'=' * 60}")
        
        #Classification_report shows per gesture precision, recall and F1
        print("\nDeatiled Classification Report:")
        print(classification_report(y_test,y_pred))
        
        #Mark the model as ready to make the predictions
        self.is_trained = True
        
        return accuracy #Returns accuracy so the caller can log it or act on it
    
    
    """
        Predicts the gesture from a single set of hand landmarks.

        Args:
            landmarks: A flat list of 63 floats — the 21 MediaPipe landmark
                       points, each with x, y, z coordinates.
                       e.g. [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]

        Returns:
            (gesture_label, confidence):
                gesture_label — the predicted gesture name, e.g. "guitar_chord"
                confidence    — probability of that prediction, 0.0 to 1.0
    """
    def predict(self, landmarks:list) -> Tuple[str, float]:
            
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() or load() first.")
        
        #Reshape from a flat list of 63 values into a 2d array of shape (1,63)
        X = np.array(landmarks).reshape(1,-1)
        
        #Then we apply the same normalisation that was used during training
        X_scaled = self.scaler.transform(X)
        
        #predict() method returns an array of labels and [0] gets the single prediction
        gesture_label= self.clf.predict(X_scaled)[0]
        
        #predict_proba() method returns a 2d array of class probability
        #np.max(...) picks the highest probability which is the highest confidence in the prediction
        confidence = float(np.max(self.clf.predict_proba(X_scaled)))
        
        return gesture_label, confidence
    
    
    """
    Saves the trained model, scaler, and gesture labels to disk.

    Args:
        filepath: Path to save the model file, e.g. "models/gesture_model.joblib"
    """
    def save(self, filepath:str):
        
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained model.")
        
        #We create the parent directory for the save if it doesnt exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Then we bundle everything the model needs into a single dictionary
        # We have to sace the scaler to because predict() must use the exact same
        # Normalisation parameters that were fitted during training
        
        model_bundle = {
            "classifier":     self.clf,            # The trained Random Forest
            "scaler":         self.scaler,         # The fitted StandardScaler
            "gesture_labels": self.gesture_labels, # The list of known gesture names
        }
        # joblib is preferred over pickle for large numpy arrays (faster, smaller files)
        joblib.dump(model_bundle, filepath)
        print(f"✓ Model saved to: {filepath}")
        
        
    """
    Loads a previously saved model from disk.

    Args:
        filepath: Path to the saved model file, e.g. "models/gesture_model.joblib"
    """
    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at: {filepath}")

        # Load the dictionary we saved in save()
        model_bundle = joblib.load(filepath)

        # Restore each component back onto this instance
        self.clf            = model_bundle["classifier"]
        self.scaler         = model_bundle["scaler"]
        self.gesture_labels = model_bundle["gesture_labels"]
        self.is_trained     = True  # Mark the model as ready to use

        print(f"✓ Model loaded from: {filepath}")
        print(f"  Gesture labels: {list(self.gesture_labels)}")
    
    
    
"""
Runs the full training pipeline:
    1. Loads all collected gesture data from disk
    2. Trains the Random Forest classifier
    3. Saves the trained model for use in recogniser.py

Run this script directly:  python src/gesture_classifier.py
"""
def main():
        classifier = GestureClassifier(model_type="random_forest")

        # Load all .pkl files from the nested instrument subfolders
        X, y = classifier.load_training_data("data/instrument_gestures")

        # Train the model and print accuracy + classification report
        accuracy = classifier.train(X, y, test_size=0.2)

        # Save the trained model to the models/ folder
        classifier.save("models/gesture_model.joblib")

        print(f"\n✓ Training complete. Final test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
        
        
        
        