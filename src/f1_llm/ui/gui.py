import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext

class FileQueryResponseUI:
    def __init__(self, master, backend):
        self.master = master
        master.title("Simple LLM Interface")

        # Upload PDF button
        self.upload_button = tk.Button(master, text="Upload PDF", command=self.upload_pdf)
        self.upload_button.pack()

        # Text entry box
        self.entry_label = tk.Label(master, text="Enter your query:")
        self.entry_label.pack()
        self.entry = tk.Entry(master, width=50)
        self.entry.pack()

        # Submit button
        self.submit_button = tk.Button(master, text="Submit", command=self.submit_query)
        self.submit_button.pack()

        # Output text area
        self.output_label = tk.Label(master, text="LLM Response:")
        self.output_label.pack()
        self.output = scrolledtext.ScrolledText(master, width=60, height=10)
        self.output.pack()

        self.backend = backend

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            self.backend.add_documents(file_path)

    def submit_query(self):
        query = self.entry.get()
        if query:
            response = self.backend.inference(query)
            self.output.delete(1.0, tk.END)
            self.output.insert(tk.END, response)

