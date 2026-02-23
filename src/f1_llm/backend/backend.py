from langchain_milvus import Milvus
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
import threading
from pymilvus import connections, utility
import re

SYSTEM_PROMPT = (
    "You are an expert designer of Formula 1 cars. "
    "Answer the user based on the following retrieved context from the regulations."
    "If the context does not contain enough information, say so. \n\n"
    "{context}"
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}")
])

VLLM_API_BASE = "http://localhost:8000/v1"


class Backend():
    def __init__(self, model_name, milvus_host, milvus_port, debug=False, fresh=False):
        self.model_name = model_name  # Store the model name
        self.debug = debug

        if "DeepSeek-R1" in model_name:
            # Connect to local vLLM server (must be running separately)
            self.llm = ChatOpenAI(
                openai_api_base=VLLM_API_BASE,
                openai_api_key="unused",
                model_name=model_name
            )
        else:
            self.llm = ChatOpenAI(name=model_name)


        # set up connection to RAG database
        embedding_function = OllamaEmbeddings(
            model='nomic-embed-text',
            query_instruction="search_query: ",
            embed_instruction="search_document: "
        )

        # Only drop collection when --fresh is passed
        connections.connect(alias="default", host=milvus_host, port=str(milvus_port))
        if fresh and utility.has_collection("test_collection"):
            utility.drop_collection("test_collection")
            print("[DEBUG] Dropped existing 'test_collection' to start fresh")

        self.vector_store = Milvus(
            embedding_function=embedding_function,
            connection_args={"host": milvus_host, "port": milvus_port},
            collection_name="test_collection",
            auto_id=True,
            primary_field="id",
            vector_field="vector",
            text_field="text"
        )
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 10}
        )

        # Build the retrieval chain
        combine_docs_chain = create_stuff_documents_chain(self.llm, RAG_PROMPT)
        self.qa_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

        # user needs to upload document
        self.documents_uploaded = True


    def inference(self, user_text):
        if not self.documents_uploaded:
            return "Please upload a document."

        chain_response = self.qa_chain.invoke({"input": user_text})
        response = chain_response["answer"]

        if self.debug:
            self._log_retrieved_context(chain_response)

        if "DeepSeek-R1" in self.model_name and "</think>" in response:
            # Parse thinking from response
            # Handle both cases: <think>...</think> or just ...</think> (vLLM may strip <think>)
            if "<think>" in response:
                before_think, remainder = response.split("<think>", 1)
                if before_think.strip():
                    yield {"type": "answer", "content": before_think}
                thought_content, answer = remainder.split("</think>", 1)
            else:
                thought_content, answer = response.split("</think>", 1)

            yield {"type": "thought_start", "content": thought_content.strip()}
            yield {"type": "thought_end", "content": thought_content.strip()}

            if answer.strip():
                yield {"type": "answer", "content": answer.strip()}
        else:
            yield {"type": "answer", "content": response}


    def _log_retrieved_context(self, chain_response):
        """Log all retrieved source documents to terminal."""
        source_docs = chain_response.get("context", [])
        print("\n" + "=" * 80)
        print(f"[DEBUG] Retrieved {len(source_docs)} context document(s)")
        print("=" * 80)
        for i, doc in enumerate(source_docs):
            print(f"\n--- Document {i + 1} ---")
            if doc.metadata:
                print(f"  Metadata: {doc.metadata}")
            print(f"  Content:")
            print(f"  {doc.page_content}")
        print("=" * 80 + "\n")


    @staticmethod
    def _inject_section_context(documents):
        """Prepend section headers to every other line so chunks carry their article context."""
        section_pattern = re.compile(r'^(\d+(?:\.\d+)+|ARTICLE\s+\d+)\s+(.*)', re.MULTILINE)
        processed_docs = []
        current_section = "Unknown Section"
        for doc in documents:
            new_lines = []
            line_count = 0
            for line in doc.page_content.split('\n'):
                match = section_pattern.match(line.strip())
                if match:
                    current_section = f"{match.group(1)} {match.group(2)}"
                if line.strip():
                    if line_count % 2 == 0:
                        new_lines.append(f"[{current_section}] {line}")
                    else:
                        new_lines.append(line)
                    line_count += 1
                else:
                    new_lines.append(line)
            doc.page_content = '\n'.join(new_lines)
            processed_docs.append(doc)
        return processed_docs

    def add_documents(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        documents = self._inject_section_context(documents)
        separators = ["\nARTICLE ", "\n\n", r"\n(?=\d+\.\d+)", "\n", " "]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1024,
            chunk_overlap = 256,
            separators=separators,
            is_separator_regex=True
        )
        splits = text_splitter.split_documents(documents)

        # Calculate total steps (number of documents)
        total_docs = len(documents)
        current_doc = 0

        # Process documents in batches
        batch_size = 10
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            _ = self.vector_store.add_documents(batch)
            
            # Update progress based on documents processed
            current_doc += len(batch)
            progress = (current_doc / len(splits)) * 100
            yield {
                "type": "upload_progress",
                "current": current_doc,
                "total": len(splits),
                "message": f"Processing document {current_doc} of {len(splits)}"
            }

        self.documents_uploaded = True
        yield {
            "type": "upload_complete",
            "message": "Document upload complete"
        }

