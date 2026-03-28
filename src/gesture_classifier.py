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
    def load_training_data(self, data_dir: str = "data/raw_landmarks") -> Tuple[np.ndarray, np.ndarray]:
        X = [] #Feature data / landmarks
        y = [] #Labels / gesture names
        
        #First we get the list of all pickle files in our data directory
        pickle_files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]
        
        if not pickle_files:
            raise FileNotFoundError(f"No gesture data found in {data_dir}")
        
        print(f"Loading training data from {data_dir}...")
        print(f"Found {len(pickle_files)} gesture file(s)")
        
        gesture_counts = {}
        
        
        #Load each gesture files data
        for pickle_file in pickle_files:
            gesture_name = pickle_file.replace('.pkl', '') #Gets the gesture name
            file_path = os.path.join(data_dir, pickle_file)
            
            try:
                with open(file_path, 'rb') as f:
                    gesture_landmarks = pickle.load(f)
                    
                    # Add all samples for this gesture
                    for landmarks in gesture_landmarks:
                        X.append(landmarks)
                        y.append(gesture_name)
                    
                    gesture_counts[gesture_name] = len(gesture_landmarks) 
                    print(f"✓ {gesture_name}: {len(gesture_landmarks)} samples")
                    
                
            except Exception as e:
                print(f"✗ Error loading {pickle_file}: {e}")
            
        
        if not X: 
            raise ValueError("No valid gestures data loaded")
        
        # Then we convert the arrays into numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        #Store gesture labels for later use
        self.gesture_labels = np.unique(y)
        
        print(f"\nTotal samples loaded: {len(X)}")
        print(f"Gesture labels: {list(self.gesture_labels)}")
        
        
        return X, y
    
    
    
    """
    This function trains the gesture classifier 
    test_size: Is the fraction of data to use for testing
    """
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        
        print("\n" + "=" * 60)
        print("Training Gesture Classifier")
        print("=" *60)
        
        
        # Split the data into training and testing sets
        print(f"\nSplitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        
        