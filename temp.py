import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def prep_real_dataset():
    os.makedirs('data', exist_ok=True)
    
    # Load the 30-dimensional continuous dataset
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    
    # Split 80% Train, 20% Test
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Compute stats ONLY on training data to prevent data leakage
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-7
    
    # Normalize
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Save to disk
    np.save('data/real_30D_train.npy', X_train_norm)
    np.save('data/real_30D_test.npy', X_test_norm)
    
    print(f"Saved Real 30D Dataset.")
    print(f"Train Shape: {X_train_norm.shape} | Test Shape: {X_test_norm.shape}")

if __name__ == "__main__":
    prep_real_dataset()