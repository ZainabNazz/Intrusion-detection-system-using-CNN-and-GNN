import os
import torch
import pickle
import numpy as np
from torch_geometric.data import Data
from models.cnn_gnn_model import CNN_GNN_Model
from sklearn.preprocessing import LabelEncoder

def load_encoder(path="encoder.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_syscall_sequence(file_path):
    with open(file_path, 'r') as f:
        line = f.read().strip()
        return list(map(int, line.split()))

def build_graph(syscalls, encoder):
    encoded = encoder.transform([sc for sc in syscalls if sc in encoder.classes_])
    if len(encoded) < 1:
        raise ValueError("No valid syscalls found in input")

    edge_index = [[i, i + 1] for i in range(len(encoded) - 1)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() \
        if edge_index else torch.empty((2, 0), dtype=torch.long)

    x = torch.nn.functional.one_hot(torch.tensor(encoded), num_classes=len(encoder.classes_)).float()
    return Data(x=x, edge_index=edge_index)

def predict(model, graph):
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    with torch.no_grad():
        out = model(graph)
        pred = out.argmax(dim=1).item()
        return "Attack" if pred == 1 else "Normal"

def main():
    encoder = load_encoder("encoder.pkl")
    input_dim = len(encoder.classes_)

    model = CNN_GNN_Model(num_features=input_dim)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Create sample test_syscalls directory if not exists
    os.makedirs("test_syscalls", exist_ok=True)

    # Example test files (auto-create if empty)
    if not os.listdir("test_syscalls"):
        print("🛠 Creating example test files...")
        examples = {
            "sample1.txt": "114 162 114 114 162 142 123 124 140 170",
            "sample2.txt": "116 117 118 116 119 121 122 117 115"
        }
        for fname, content in examples.items():
            with open(f"test_syscalls/{fname}", "w") as f:
                f.write(content)

    print("🔍 Predicting samples in test_syscalls/ ...\n")
    for filename in os.listdir("test_syscalls"):
        if filename.endswith(".txt"):
            path = os.path.join("test_syscalls", filename)
            try:
                syscalls = load_syscall_sequence(path)
                graph = build_graph(syscalls, encoder)
                label = predict(model, graph)
                print(f"{filename}: {label}")
            except Exception as e:
                print(f"⚠️ {filename}: Failed to predict — {e}")

if __name__ == "__main__":
    main()
