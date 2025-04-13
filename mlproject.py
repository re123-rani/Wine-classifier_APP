import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

# Load dataset
file_path = "C:/Users/Lenovo/Downloads/WineQT.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# Add ID column if not present
if 'Id' not in df.columns:
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Id'}, inplace=True)

# Preprocessing
df.fillna(df.mean(), inplace=True)
df['best quality'] = (df['quality'] > 5).astype(int)
features = df.drop(columns=['quality', 'best quality'])
target = df['best quality']
feature_ranges = {col: (features[col].min(), features[col].max()) for col in features.columns if col != 'Id'}

xtrain, xtest, ytrain, ytest = train_test_split(
    features.drop(columns=['Id']), target, test_size=0.2, random_state=42, stratify=target)

imputer = SimpleImputer(strategy='mean')
xtrain = imputer.fit_transform(xtrain)
xtest = imputer.transform(xtest)

scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

model = XGBClassifier(eval_metric="logloss")
model.fit(xtrain, ytrain)

# GUI
root = tk.Tk()
root.title("🍷 Wine Classifier App")
root.geometry("650x850")
root.configure(bg="#34495e")

font_label = ("Segoe UI", 11, "bold")
font_entry = ("Segoe UI", 10)
font_button = ("Segoe UI", 11, "bold")
font_hint = ("Segoe UI", 9, "italic")

# Title
title_label = tk.Label(root, text="🍷 Wine Classifier App", font=("Segoe UI", 18, "bold"), bg="#34495e", fg="#ecf0f1")
title_label.pack(pady=15)

# ID Frame
id_frame = tk.Frame(root, bg="#34495e")
id_frame.pack(pady=5)
tk.Label(id_frame, text="Search by Wine ID:", font=font_label, bg="#34495e", fg="white").grid(row=0, column=0, padx=5)
id_entry = tk.Entry(id_frame, font=font_entry, width=12)
id_entry.grid(row=0, column=1, padx=5)
search_button = tk.Button(id_frame, text="🔍 Search", command=lambda: search_by_id(), font=font_button, bg="#2980b9", fg="white", width=10)
search_button.grid(row=0, column=2, padx=5)

# Input Frame
frame = tk.Frame(root, bg="#ecf0f1", padx=20, pady=15, relief="raised", borderwidth=3)
frame.pack(pady=10)

entries = []
labels = [col for col in features.columns if col != 'Id']

def title_case(text):
    return ' '.join(word.capitalize() for word in text.split())

for i, label in enumerate(labels):
    min_val, max_val = feature_ranges[label]
    range_hint = f"(Range: {min_val:.2f} - {max_val:.2f})"
    
    tk.Label(frame, text=title_case(label), font=font_label, bg="#ecf0f1").grid(row=i, column=0, padx=10, pady=6, sticky="w")
    tk.Label(frame, text=range_hint, font=font_hint, fg="#c0392b").grid(row=i, column=1, sticky="w")
    entry = tk.Entry(frame, font=font_entry, width=20, relief="solid", borderwidth=1, bg="white")
    entry.grid(row=i, column=2, padx=10, pady=6)
    entries.append(entry)

# Result
result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, font=("Segoe UI", 14, "bold"), fg="#ecf0f1", bg="#34495e")
result_label.pack(pady=10)

# Buttons
button_frame = tk.Frame(root, bg="#34495e")
button_frame.pack(pady=10)

predict_button = tk.Button(button_frame, text="✅ Predict Quality", command=lambda: predict_quality(), font=font_button,
                           bg="#27ae60", fg="white", relief="raised", padx=10, pady=5, width=20)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="🗑 Clear", command=lambda: clear_fields(), font=font_button,
                         bg="#c0392b", fg="white", relief="raised", padx=10, pady=5, width=20)
clear_button.grid(row=0, column=1, padx=10)

# Functions
def predict_quality():
    try:
        user_data = [float(entry.get()) for entry in entries]
        user_data = np.array(user_data).reshape(1, -1)
        user_data = scaler.transform(user_data)
        prediction = model.predict(user_data)[0]
        if prediction == 1:
            result_text.set("✅ Good Quality Wine (1)")
            result_label.config(fg="#2ecc71")
            messagebox.showinfo("✅ Drinkable Wine", "This wine is good quality. You can drink it!")
        else:
            result_text.set("❌ Bad Quality Wine (0)")
            result_label.config(fg="#e74c3c")
            messagebox.showwarning("❗ Health Warning", "⚠️ This wine is not considered healthy!")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

def clear_fields():
    for entry in entries:
        entry.delete(0, tk.END)
    result_text.set("")
    id_entry.delete(0, tk.END)

def search_by_id():
    try:
        wine_id = int(id_entry.get())
        wine_row = df[df['Id'] == wine_id]
        if wine_row.empty:
            messagebox.showerror("Not Found", f"No wine found with ID {wine_id}")
            return
        feature_values = wine_row[features.columns[features.columns != 'Id']].iloc[0].to_list()
        for i, value in enumerate(feature_values):
            entries[i].delete(0, tk.END)
            entries[i].insert(0, str(value))
    except ValueError:
        messagebox.showerror("Input Error", "Please enter a valid Wine ID (integer).")

# Launch GUI
root.mainloop()
