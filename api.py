from flask import Flask, request, jsonify
import torch
import os
import pickle
from werkzeug.utils import secure_filename
from models.cnn_gnn_model import CNN_GNN_Model
from torch_geometric.data import Data
import torch.nn.functional as F

app = Flask(__name__)
UPLOAD_FOLDER = 'test_syscalls'
ALLOWED_EXTENSIONS = {'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load encoder and model
encoder = pickle.load(open("encoder.pkl", "rb"))
input_dim = len(encoder.classes_)
model = CNN_GNN_Model(num_features=input_dim)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def build_graph(syscalls, encoder):
    # Only keep valid syscalls
    valid_syscalls = [sc for sc in syscalls if sc in encoder.classes_]
    if not valid_syscalls:
        raise ValueError("No valid syscalls found in the input.")
    encoded = encoder.transform(valid_syscalls)
    edge_index = [[i, i + 1] for i in range(len(encoded) - 1)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() \
        if edge_index else torch.empty((2, 0), dtype=torch.long)
    x = F.one_hot(torch.tensor(encoded), num_classes=len(encoder.classes_)).float()
    return Data(x=x, edge_index=edge_index)

@app.route('/predict', methods=['POST'])
def predict_syscall_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        try:
            with open(path, 'r') as f:
                syscalls = list(map(int, f.read().strip().split()))
            graph = build_graph(syscalls, encoder)
            graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
            with torch.no_grad():
                out = model(graph)
                pred = out.argmax(dim=1).item()
                confidence = F.softmax(out, dim=1)[0][pred].item()
                label = "Attack" if pred == 1 else "Normal"
            return jsonify({
                'filename': filename,
                'prediction': label,
                'confidence': round(confidence, 4)
            })
        except Exception as e:
            return jsonify({'error': f'Failed to process file: {str(e)}'}), 500
    return jsonify({'error': 'Invalid file type. Only .txt allowed.'}), 400

@app.route('/')
def index():
    return jsonify({
        'message': 'Welcome to the Intrusion Detection System API.',
        'usage': 'Send a POST request to /predict with a .txt file of system calls.'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
