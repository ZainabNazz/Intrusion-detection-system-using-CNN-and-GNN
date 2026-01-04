import torch
import pickle
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models.cnn_gnn_model import CNN_GNN_Model
from utils.data_preprocessing import prepare_dataset

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Load dataset
print("📦 Loading dataset...")
graphs = prepare_dataset()  # No limit means use entire dataset

# Extract input dimension
input_dim = graphs[0].x.shape[1]
print(f"🎯 Feature dimension: {input_dim}")

# Split dataset
train_data, val_data = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# Define model
model = CNN_GNN_Model(num_features=input_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# TensorBoard (optional)
writer = SummaryWriter("runs/ids_train")

# Training loop
print("🚀 Starting training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"📈 Epoch {epoch}: Train Loss = {avg_loss:.4f}")
    writer.add_scalar("Loss/train", avg_loss, epoch)

# Save model
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved as model.pth")

# Save encoder (already done in prepare_dataset(), this is a reminder)
print("✅ Encoder saved as encoder.pkl")
writer.close()