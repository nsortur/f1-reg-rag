# f1-reg-rag

![app-logo](thumbnail.png)

## Local RAG development

This project is a RAG-based ideation tool for Formula 1 technical regulations. It allows users to ask questions and critically evaluate potential design decisions, focusing on removing false positives and hallucinations. It also lowers the barrier to entry for fans to act as designers – are you the next Adrian Newey?

Set-up instructions:

1. Set up local Milvus database and Ollama in Docker container
2. Run `src/utils/setup_local_milvus.py` to add collection and vector size (uses `nomic-embed-text` embedding)
3. Set env `OPENAI_API_KEY` if using an OpenAI LLM when running the application
4. Set up vLLM server (if using DeepSeek-R1) with PagedAttention. The parameters I used on an RTX 3090 are below:

```
vllm serve neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 4
```
5. Run `python src/f1_llm/main.py --llm <llm>`
    - gpt-4o-mini
    - neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16
6. Additionally, you can add `--debug` to log all retrieved RAG context to terminal and `--fresh` to drop existing collection and re-embed regulation documents from scratch.

## Blender integration

There is an additional `--blender_mcp` flag, which allows you to use Blender MCP to execute Python code in Blender. This is useful for visualizing F1 components based on the regulations.

1. Start the standalone blender server with `blender -b -P src/f1_llm/utils/blender_worker.py`
- This registers the addon class and starts the socket server on port 9876 (addon from [Blender MCP](https://github.com/ahujasid/blender-mcp/tree/main))


## System requirements

This project uses the 4-bit quantized 32 billion parameter model of DeepSeek-R1. System will need >= 20GB vRAM to locally host the model, which can be accomodated with an RTX 3090, 4090, or 5090 (or larger).
