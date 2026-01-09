# f1-reg-rag

![app-logo](thumbnail.png)

Set-up instructions:

1. Set up local Milvus database and Ollama in Docker container
2. Run `src/utils/setup_local_milvus.py` to add collection and vector size (uses DeepSeek-R1 embedding)
3. Set env `OPENAI_API_KEY` if using an OpenAI LLM when running the application
4. Run `python src/f1_llm/main.py --llm <llm>`
    - gpt-4o-mini
    - neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16

## System requirements

This project uses the 4-bit quantized 32 billion parameter model of DeepSeek-R1. System will need >= 20GB vRAM to locally host the model, which can be accomodated with an RTX 3090, 4090, or 5090 (or larger).
