from ui.gui import FileQueryResponseUI
from  backend.backend import Backend
import tkinter as tk
import argparse

def main(llm):
    milvus_host = "localhost"
    milvus_port = 19530
    backend = Backend(model_name=llm, milvus_host=milvus_host, milvus_port=milvus_port)

    root = tk.Tk()
    app = FileQueryResponseUI(root, backend)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the File Query Response UI with specified LLM model.")
    parser.add_argument('-llm', choices=['gpt-4o-mini', 
                                         'neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16'], 
                                         default='gpt-4o-mini', help='Choose the LLM model to use.')
    args = parser.parse_args()

    main(args.llm)
