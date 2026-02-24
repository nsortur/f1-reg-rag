from ui.gui import FileQueryResponseUI
from  backend.backend import Backend
import tkinter as tk
import argparse

# DEBUG_PDF_PATH = "/home/nsortur/f1-llm/f1_2025_regs_engine_5p.pdf"
DEBUG_PDF_PATH = "/home/nsortur/f1-llm/full_2026_regs.pdf"

def main(llm, debug=False, fresh=False, blender_mcp=False):
    milvus_host = "localhost"
    milvus_port = 19530
    backend = Backend(model_name=llm, milvus_host=milvus_host, milvus_port=milvus_port, debug=debug, fresh=fresh, blender_mcp=blender_mcp)

    if debug and fresh:
        print(f"[DEBUG] Auto-loading PDF: {DEBUG_PDF_PATH}")
        for progress in backend.add_documents(DEBUG_PDF_PATH):
            print(f"[DEBUG] {progress['message']}")

    root = tk.Tk()
    app = FileQueryResponseUI(root, backend)
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the File Query Response UI with specified LLM model.")
    parser.add_argument('--llm', choices=['gpt-4o-mini', 
                                         'neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16',
                                         'o3-mini',
                                         'o3'], 
                                         default='gpt-4o-mini', help='Choose the LLM model to use.')
    parser.add_argument('--debug', action='store_true', help='Log all retrieved RAG context to terminal.')
    parser.add_argument('--fresh', action='store_true', help='Drop existing collection and re-embed documents from scratch.')
    parser.add_argument('--blender_mcp', action='store_true', help='Enable Blender MCP tool for visualization.')
    args = parser.parse_args()

    main(args.llm, debug=args.debug, fresh=args.fresh, blender_mcp=args.blender_mcp)
