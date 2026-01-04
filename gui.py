import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import requests
import os
import csv
import threading
from datetime import datetime
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

API_URL = "http://localhost:5000/predict"
DB_PATH = "ids_logs.db"

class IDSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔐 Intrusion Detection")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        self.root.configure(bg="#2c3e50")

        self.history = []
        self.create_database()

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview",
                        background="#ecf0f1",
                        foreground="black",
                        rowheight=25,
                        fieldbackground="#ecf0f1")
        style.configure("Treeview.Heading", background="#2980b9", foreground="white")

        self.title_label = tk.Label(root, text="🔐 Intrusion Detection System", font=("Arial", 20, "bold"),
                                    bg="#2c3e50", fg="white")
        self.title_label.pack(pady=15)

        btn_frame = tk.Frame(root, bg="#2c3e50")
        btn_frame.pack(pady=5)

        self.upload_btn = tk.Button(btn_frame, text="📂 Upload File", command=self.upload_file,
                                    font=("Arial", 12), bg="#1abc9c", fg="white", width=20)
        self.upload_btn.grid(row=0, column=0, padx=5)

        self.live_btn = tk.Button(btn_frame, text="🔁 Live Monitor", command=self.start_live_monitoring,
                                  font=("Arial", 12), bg="#3498db", fg="white", width=20)
        self.live_btn.grid(row=0, column=1, padx=5)

        self.export_btn = tk.Button(btn_frame, text="💾 Export CSV", command=self.export_to_csv,
                                    font=("Arial", 12), bg="#f39c12", fg="white", width=20)
        self.export_btn.grid(row=0, column=2, padx=5)

        self.result_label = tk.Label(root, text="", font=("Arial", 13), bg="#2c3e50", fg="white")
        self.result_label.pack(pady=5)

        self.confidence_label = tk.Label(root, text="", font=("Arial", 11, "italic"), bg="#2c3e50", fg="lightgray")
        self.confidence_label.pack()

        self.tree = ttk.Treeview(root, columns=("File", "Prediction", "Confidence"), show="headings", height=6)
        self.tree.heading("File", text="File")
        self.tree.heading("Prediction", text="Prediction")
        self.tree.heading("Confidence", text="Confidence")
        self.tree.pack(pady=10)

        graph_btn = tk.Button(root, text="📊 Show Prediction Graph", command=self.plot_graph,
                              font=("Arial", 12), bg="#8e44ad", fg="white")
        graph_btn.pack(pady=10)

    def create_database(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log_to_db(self, filename, prediction, confidence):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO logs (filename, prediction, confidence, timestamp) VALUES (?, ?, ?, ?)",
                       (filename, prediction, confidence, timestamp))
        conn.commit()
        conn.close()

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            self.send_to_api(file_path)

    def send_to_api(self, file_path):
        try:
            with open(file_path, 'rb') as file:
                files = {'file': file}
                response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction", "Unknown")
                confidence = data.get("confidence", 0)
                fname = os.path.basename(file_path)

                self.result_label.config(
                    text=f"Prediction: {prediction}", fg="#2ecc71" if prediction == "Normal" else "#e74c3c")
                self.confidence_label.config(
                    text=f"Confidence: {confidence * 100:.2f}%")

                self.tree.insert('', 'end', values=(fname, prediction, f"{confidence * 100:.2f}%"))
                self.history.append((fname, prediction, confidence))
                self.log_to_db(fname, prediction, confidence)

                self.root.after(100, lambda: messagebox.showinfo("Detection", f"{fname}: {prediction} ({confidence * 100:.2f}%)"))

            else:
                error_msg = response.json().get("error", "Unknown error occurred.")
                self.result_label.config(text=f"❌ Error: {error_msg}", fg="red")
                self.confidence_label.config(text="")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to send request: {str(e)}")

    def export_to_csv(self):
        if not self.history:
            messagebox.showwarning("No Data", "No predictions to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "Prediction", "Confidence"])
            for row in self.history:
                writer.writerow([row[0], row[1], f"{row[2] * 100:.2f}%"])
        messagebox.showinfo("Export Successful", f"Results exported to {file_path}")

    def start_live_monitoring(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        messagebox.showinfo("Live Monitoring", f"Monitoring {folder_path} for .txt files...")
        threading.Thread(target=self.monitor_folder, args=(folder_path,), daemon=True).start()

    def monitor_folder(self, folder_path):
        seen_files = set()
        while True:
            current_files = set(f for f in os.listdir(folder_path) if f.endswith(".txt"))
            new_files = current_files - seen_files
            for fname in new_files:
                full_path = os.path.join(folder_path, fname)
                self.send_to_api(full_path)
                seen_files.add(fname)
            self.root.after(3000)

    def plot_graph(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT prediction, COUNT(*) FROM logs GROUP BY prediction")
        data = cursor.fetchall()
        conn.close()

        labels = [row[0] for row in data]
        values = [row[1] for row in data]

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(labels, values, color=["#2ecc71" if l == "Normal" else "#e74c3c" for l in labels])
        ax.set_title("Prediction Distribution")
        ax.set_ylabel("Count")

        top = tk.Toplevel(self.root)
        top.title("Prediction Graph")
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack()


def main():
    root = tk.Tk()
    app = IDSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()