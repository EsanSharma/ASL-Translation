import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import (
    ACTIONS, NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES, 
    BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_DIR
)
from src.model import ASLGestureClassifier
from src.dataset import ASLDataset, load_data

def train_model():
    print("Loading landmark data...")
    X, y = load_data()
    
    if len(X) == 0:
        print("No training data found in data/processed/. Run src/collect_data.py first.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(ASLDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ASLDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLGestureClassifier(NUM_FEATURES, HIDDEN_SIZE, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0

    for epoch in range(1, 81):
        model.train()
        train_loss, train_correct = 0.0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_correct += (outputs.argmax(1) == batch_y).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_correct += (outputs.argmax(1) == batch_y).sum().item()

        val_acc = val_correct / len(X_val)
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/80] | Train Acc: {(train_correct/len(X_train))*100:.1f}% | Val Acc: {val_acc*100:.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")

    print(f"\n[SUCCESS] Model trained. Peak Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == "__main__":
    train_model()