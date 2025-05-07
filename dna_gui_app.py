import tkinter as tk
from tkinter import messagebox
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model saved in .keras format
model = load_model("dna_cnn_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Parameters
KMER_SIZE = 6
MAX_LEN = 500

# Preprocessing functions
def clean_sequence(seq):
    return re.sub(r'[^ACGT]', '', seq.upper())

def kmer_tokenizer(seq, k=KMER_SIZE):
    return ' '.join([seq[i:i+k] for i in range(len(seq) - k + 1)])

def preprocess_sequence(seq):
    seq = clean_sequence(seq)
    kmers = kmer_tokenizer(seq)
    tokenized = tokenizer.texts_to_sequences([kmers])
    padded = pad_sequences(tokenized, maxlen=MAX_LEN)
    return padded

# Prediction function
def classify_sequence():
    seq = entry.get("1.0", tk.END).strip()
    if len(seq) < KMER_SIZE:
        messagebox.showerror("Error", f"Sequence must be at least {KMER_SIZE} bases long.")
        return
    
    input_data = preprocess_sequence(seq)
    probs = model.predict(input_data)[0]
    pred_index = np.argmax(probs)
    pred_label = label_encoder.inverse_transform([pred_index])[0]

    result = f"Prediction: {pred_label.upper()}\n\nConfidence:\n"
    for i, label in enumerate(label_encoder.classes_):
        result += f"{label.capitalize()}: {probs[i]:.2%}\n"

    result_label.config(text=result)

# GUI Setup
root = tk.Tk()
root.title("DNA Sequence Classifier - Deep Learning")

tk.Label(root, text="Enter DNA Sequence:", font=("Arial", 14)).pack(pady=10)
entry = tk.Text(root, height=6, width=60, font=("Courier", 12))
entry.pack()

tk.Button(root, text="Classify", command=classify_sequence, font=("Arial", 12), bg="#4CAF50", fg="white").pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 12), justify="left")
result_label.pack(pady=10)

root.mainloop()
