import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from utils.data_preprocessing import prepare_dataset
from models.cnn_gnn_model import CNN_GNN_Model

def evaluate():
    print("📦 Loading dataset for evaluation...")
    dataset = prepare_dataset()
    input_dim = dataset[0].x.shape[1]

    print("⚙️ Loading trained model...")
    model = CNN_GNN_Model(num_features=input_dim)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Load encoder
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    # Split into validation set
    _, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_loader = DataLoader(val_data, batch_size=32)

    y_true = []
    y_pred = []
    y_scores = []

    print("🔍 Running inference...")
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1]  # Probability of attack
            preds = torch.argmax(out, dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            y_true.extend(batch.y.cpu().numpy())

    # 📊 Confusion Matrix
    print("\n📊 Evaluation Results")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Confusion matrix saved as confusion_matrix.png")

    # 🔢 Evaluation Metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0
    adr = tp / (tp + fn) if (tp + fn) else 0

    print(f"F1-Score            : {f1:.2f}")
    print(f"Precision           : {precision:.2f}")
    print(f"Recall              : {recall:.2f}")
    print(f"False Positive Rate : {fpr:.2f}")
    print(f"Attack Detection Rate: {adr:.2f}")

    # 📝 Save text report
    with open("metrics_report.txt", "w") as f:
        f.write(f"Accuracy : {accuracy:.2f}")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"False Positive Rate: {fpr:.4f}\n")
        f.write(f"Attack Detection Rate: {adr:.4f}\n")
    print("📝 Metrics report saved as metrics_report.txt")

    # 📉 ROC Curve with AUC and annotation
    fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_curve, tpr_curve)

    plt.figure()
    plt.plot(fpr_curve, tpr_curve, color='blue', label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.scatter(fpr, adr, color='red', label="Operating Point")
    plt.text(fpr + 0.02, adr, f"FPR={fpr:.2f}\nADR={adr:.2f}", fontsize=9)
    plt.xlabel("False Positive Rate")
    plt.ylabel("Attack Detection Rate (TPR)")
    plt.title("ROC Curve: Attack Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    print("📉 ROC curve saved as roc_curve.png")

    # 🔘 Single-point Detection Rate vs FPR
    plt.figure()
    plt.plot([fpr], [adr], marker='o', color='red')
    plt.subplots_adjust(left=0.1, right=0.9)  # Manually tweak margins
    plt.text(fpr + 0.01, adr, f"FPR={fpr:.2f}\nADR={adr:.2f}", fontsize=10, color='black')
    plt.xlabel("False Positive Rate")
    plt.ylabel("Attack Detection Rate")
    plt.title("Attack Detection Rate vs False Positive Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("detection_vs_fpr.png")
    print("📈 Point plot saved as detection_vs_fpr.png")

if __name__ == "__main__":
    evaluate()