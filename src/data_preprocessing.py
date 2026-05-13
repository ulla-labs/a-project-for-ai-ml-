import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_and_preprocess(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Basic cleaning
    df = df.dropna()
    
    # Feature/Target split (assuming last column is target)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=32):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, test_loader

if __name__ == "__main__":
    print("Preprocessing script initialized.")
    # Usage: X_tr, X_te, y_tr, y_te = load_and_preprocess('data/raw/dataset.csv')
