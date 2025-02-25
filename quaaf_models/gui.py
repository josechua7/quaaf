import tkinter as tk

def process_input():
    user_input = entry.get()
    result = f"You entered: {user_input}"
    result_label.config(text=result)

# Create the main window
root = tk.Tk()
root.title("Input App")

# Create a label, entry, and button
tk.Label(root, text="Enter your input:").pack(pady=10)
entry = tk.Entry(root, width=40)
entry.pack(pady=5)
tk.Button(root, text="Submit", command=process_input).pack(pady=10)
result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()