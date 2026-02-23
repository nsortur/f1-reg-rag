import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext
import threading
from tkinter import ttk

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

        # Progress bar
        self.progress_frame = tk.Frame(master)
        self.progress_frame.pack(fill=tk.X, padx=5, pady=5)
        self.progress_label = tk.Label(self.progress_frame, text="Progress:")
        self.progress_label.pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, padx=5)
        self.progress_text = tk.Label(self.progress_frame, text="0%")
        self.progress_text.pack(side=tk.LEFT)

        # Output text area
        self.output_label = tk.Label(master, text="LLM Response:")
        self.output_label.pack()
        self.output = scrolledtext.ScrolledText(master, width=60, height=10)
        self.output.pack()

        # Thoughts text area
        self.thoughts_label = tk.Label(master, text="Thought Process:")
        self.thoughts_label.pack()
        self.thoughts = scrolledtext.ScrolledText(master, width=60, height=5)
        self.thoughts.pack()

        self.backend = backend
        self.current_thought = ""
        self.thought_count = 0
        self.is_processing = False
        self.total_chunks = 0
        self.processed_chunks = 0

    def update_upload_progress(self, progress_info):
        if progress_info["type"] == "upload_progress":
            progress = (progress_info["current"] / progress_info["total"]) * 100
            self.progress_bar['value'] = progress
            self.progress_text.config(text=f"{int(progress)}% - {progress_info['message']}")
        elif progress_info["type"] == "upload_complete":
            self.progress_text.config(text="Upload complete!")
            self.master.after(2000, self.hide_upload_progress)  # Hide after 2 seconds
        self.master.update()

    def hide_upload_progress(self):
        self.progress_bar['value'] = 0
        self.progress_text.config(text="0%")

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if file_path:
            # Reset progress
            self.progress_bar['value'] = 0
            self.progress_text.config(text="0%")
            
            # Process upload and update progress
            for progress_info in self.backend.add_documents(file_path):
                self.master.after(0, self.update_upload_progress, progress_info)

    def update_progress(self):
        if self.total_chunks > 0:
            progress = (self.processed_chunks / self.total_chunks) * 100
            self.progress_bar['value'] = progress
            self.progress_text.config(text=f"{int(progress)}%")
            self.master.update()

    def update_ui(self, chunk):
        if chunk["type"] == "thought_start":
            self.current_thought = chunk["content"]
            self.thought_count += 1
            self.thoughts.insert(tk.END, f"Thought {self.thought_count}:\n{chunk['content']}")
        elif chunk["type"] == "thought_end":
            self.thoughts.insert(tk.END, "\n\n")
        elif chunk["type"] == "answer":
            self.output.insert(tk.END, chunk["content"])
        
        # Update progress
        self.processed_chunks += 1
        self.update_progress()
        
        # Update the UI
        self.master.update()

    def process_stream(self, query):
        try:
            # Reset progress
            self.total_chunks = 0
            self.processed_chunks = 0
            self.progress_bar['value'] = 0
            self.progress_text.config(text="0%")
            
            # Single pass: collect all chunks first
            chunks = list(self.backend.inference(query))
            self.total_chunks = len(chunks)
            
            # Process the collected chunks
            for chunk in chunks:
                self.master.after(0, self.update_ui, chunk)
        finally:
            self.is_processing = False
            self.submit_button.config(state='normal')
            # Ensure progress bar shows 100% when complete
            self.progress_bar['value'] = 100
            self.progress_text.config(text="100%")

    def submit_query(self):
        if self.is_processing:
            return

        query = self.entry.get()
        if query:
            # Clear previous content
            self.output.delete(1.0, tk.END)
            self.thoughts.delete(1.0, tk.END)
            
            # Reset state
            self.current_thought = ""
            self.thought_count = 0
            self.is_processing = True
            
            # Disable submit button while processing
            self.submit_button.config(state='disabled')
            
            # Start processing in a separate thread
            thread = threading.Thread(target=self.process_stream, args=(query,))
            thread.daemon = True
            thread.start()

