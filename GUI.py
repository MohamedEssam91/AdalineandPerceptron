import tkinter as tk
from tkinter import ttk
from Adaline_Model import *
from Perceptron_Model import *

def submit():
    print("Feature 1: ", combo_feature1.get())
    print("Feature 2: ", combo_feature2.get())
    print("Class 1: ", combo_class1.get())
    print("Class 2: ", combo_class2.get())
    print("Learning Rate (eta): ", entry_eta.get())
    print("Number of Epochs (m): ", entry_epochs.get())
    print("MSE Threshold: ", entry_mse_threshold.get())
    print("Add Bias: ", add_bias_var.get())
    print("Algorithm: ", algorithm.get())
    if algorithm.get() == "Adaline":
        Adaline_GUI(combo_class1.get(), combo_class2.get(), combo_feature1.get(), combo_feature2.get(), float(entry_eta.get()),  int(entry_epochs.get()), float(entry_mse_threshold.get()), bool(add_bias_var.get()))
    else:
        Perceptron_GUI(combo_class1.get(), combo_class2.get(), combo_feature1.get(), combo_feature2.get(), float(entry_eta.get()),  int(entry_epochs.get()), float(entry_mse_threshold.get()), bool(add_bias_var.get()))

root = tk.Tk()
root.title("GUI for Classification")

# Features
features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
classes = ["C1", "C2", "C3"]

# Labels
label_feature1 = ttk.Label(root, text="Select Feature 1:")
label_feature1.grid(row=0, column=0, padx=5, pady=5)

label_feature2 = ttk.Label(root, text="Select Feature 2:")
label_feature2.grid(row=1, column=0, padx=5, pady=5)

label_class1 = ttk.Label(root, text="Select Class 1:")
label_class1.grid(row=2, column=0, padx=5, pady=5)

label_class2 = ttk.Label(root, text="Select Class 2:")
label_class2.grid(row=3, column=0, padx=5, pady=5)

label_eta = ttk.Label(root, text="Enter Learning Rate (eta):")
label_eta.grid(row=4, column=0, padx=5, pady=5)

label_epochs = ttk.Label(root, text="Enter Number of Epochs (m):")
label_epochs.grid(row=5, column=0, padx=5, pady=5)

label_mse_threshold = ttk.Label(root, text="Enter MSE Threshold:")
label_mse_threshold.grid(row=6, column=0, padx=5, pady=5)

# Combo boxes
combo_feature1 = ttk.Combobox(root, values=features)
combo_feature1.grid(row=0, column=1, padx=5, pady=5)
combo_feature1.current(0)

combo_feature2 = ttk.Combobox(root, values=features)
combo_feature2.grid(row=1, column=1, padx=5, pady=5)
combo_feature2.current(1)

combo_class1 = ttk.Combobox(root, values=classes)
combo_class1.grid(row=2, column=1, padx=5, pady=5)
combo_class1.current(0)

combo_class2 = ttk.Combobox(root, values=classes)
combo_class2.grid(row=3, column=1, padx=5, pady=5)
combo_class2.current(1)

# Entries
entry_eta = ttk.Entry(root)
entry_eta.grid(row=4, column=1, padx=5, pady=5)

entry_epochs = ttk.Entry(root)
entry_epochs.grid(row=5, column=1, padx=5, pady=5)

entry_mse_threshold = ttk.Entry(root)
entry_mse_threshold.grid(row=6, column=1, padx=5, pady=5)

# Checkbox
add_bias_var = tk.IntVar()
check_bias = ttk.Checkbutton(root, text="Add Bias", variable=add_bias_var)
check_bias.grid(row=7, column=0, padx=5, pady=5)

# Radio buttons
algorithm = tk.StringVar()
radio_perceptron = ttk.Radiobutton(root, text="Perceptron", variable=algorithm, value="Perceptron")
radio_perceptron.grid(row=8, column=0, padx=5, pady=5)

radio_adaline = ttk.Radiobutton(root, text="Adaline", variable=algorithm, value="Adaline")
radio_adaline.grid(row=8, column=1, padx=5, pady=5)

# Button
submit_button = ttk.Button(root, text="Submit", command=submit)
submit_button.grid(row=9, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()