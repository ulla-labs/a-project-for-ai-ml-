import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import SimpleMLP
from data_preprocessing import create_dataloaders

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

def main():
    # Hyperparameters
    INPUT_DIM = 10
    HIDDEN_DIM = 64
    OUTPUT_DIM = 2
    LR = 0.001
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize components
    model = SimpleMLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting training on {DEVICE}...")
    # Note: Replace with actual data loaders in production
    for epoch in range(EPOCHS):
        # train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")

if __name__ == "__main__":
    main()