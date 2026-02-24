from langchain_milvus import Milvus
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import threading
from pymilvus import connections, utility
import re

# --- Model type detection ---
O3_MODELS = {"o3", "o3-mini"}

def _model_type(model_name):
    """Classify model into 'deepseek', 'o3', or 'standard'."""
    if "DeepSeek-R1" in model_name:
        return "deepseek"
    if model_name in O3_MODELS:
        return "o3"
    return "standard"

SIMPLE_RAG_PROMPT = (
    "You are an expert on F1 Technical Regulations. "
    "Your goal is to provide clear, accurate, and concise answers based on the provided context "
    "and following your intuition, as long as your intuition follows the rules of the context exactly.\n\n"
    "GUIDELINES:\n"
    "- If the context contains the answer, cite the Article number (e.g., 'Per Article 2.8.2...').\n"
    "- If the information is missing, clearly state that the regulations do not specify it.\n"
    "- Maintain a professional, technical tone.\n\n"
    "CONTEXT:\n"
    "{context}"
)

def _build_rag_chat_prompt(model_name):
    """Build the RAG chat prompt template."""
    return ChatPromptTemplate.from_messages([
        ("system", SIMPLE_RAG_PROMPT),
        ("human", "{input}")
    ])

SYSTEM_PROMPT = (
    "You are an F1 Aerodynamicist. Use the following Technical Regulations to design components.\n\n"
    "If the user asks about something that is not related to F1 design or the regulations, politely decline to answer."
    
    "### 1. MATHEMATICAL CONSTRAINTS (CRITICAL)\n"
    "- COORDINATES: Right-handed Cartesian (X, Y, Z).\n"
    "  - X: Longitudinal. X=0 at Front of Survival Cell (XA=0). INCREASES REARWARDS.\n"
    "  - Y: Transverse. Y=0 is Symmetry Plane. INCREASES TO DRIVER'S RIGHT.\n"
    "  - Z: Vertical. Z=0 is the Reference Plane (sprung floor). INCREASES UPWARDS.\n"
    "- UNIT CONVERSION: Regulations use Millimeters (mm). Blender uses Meters (m).\n"
    "  - YOU MUST CONVERT ALL VALUES: (Value in mm / 1000 = Value in Blender).\n"
    "- SYMMETRY: Rules are defined for +Y. Always mirror or duplicate for -Y unless told otherwise.\n\n"

    "### 2. PRINCIPAL PLANES (DATUMS)\n"
    "- XA=0: Forward limit of Survival Cell.\n"
    "- XF=0: Front Wheel Centerline (Between XA=0 and XA=150).\n"
    "- XR=0: Rear Wheel Centerline (Wheelbase <= 3400mm from XF=0).\n"
    "- Z=0: Bottom of the sprung part of the car.\n\n"

    "### 3. DESIGN PIPELINE\n"
    "1. Identify the requested component.\n"
    "2. Locate the specific Article in the context below.\n"
    "3. MANDATORY: Call 'get_scene_info' FIRST, before writing ANY code. "
    "Use it to find the locations, dimensions, and names of existing objects. "
    "You MUST base your coordinates on these results. NEVER hardcode or guess positions.\n"
    "4. Calculate the bounding box using the X, Y, Z logic above. \n"
    "If an Article refers to local coordinates (XW, YW, ZW), translate them to the global car system (X, Y, Z) before drawing.\n"
    "5. Generate 'bpy' code to use the 'execute_blender_visualization' tool. ALWAYS use a 'Delete-and-Rebuild' strategy. "
    "Delete-and-Rebuild Strategy: To avoid name collisions (e.g., 'Wing.001'), your code must explicitly check for and "
    "delete existing objects by name at the start of the script. "
    "Do not attempt to 'edit' or 'move' existing meshes; it is more reliable to "
    "re-create the entire component hierarchy from scratch with updated parameters."

    "### 4. REGULATION SOURCE TEXT (FOR REFERENCE):\n"
    "{context}"
)

VLLM_API_BASE = "http://localhost:8000/v1"


def _setup_blender_tools():
    """Lazy setup of the Blender tools."""
    from langchain.tools import tool
    from .blender_client import BlenderMCPClient

    client = BlenderMCPClient()

    @tool
    def execute_blender_visualization(python_code: str):
        """
        Executes Python code in Blender. Use this to create or move F1 components.
        Example: 'import bpy; bpy.ops.mesh.primitive_cube_add(size=2)'
        """
        clean_code = python_code.strip().replace("```python", "").replace("```", "")
        response = client.send_command("execute_code", {"code": clean_code})
        if response.get("status") == "success":
            return f"Visualization updated. Result: {response.get('result', 'Done')}"
        return f"Error visualizing: {response.get('message')}"

    @tool
    def get_scene_info():
        """
        Returns a list of all objects in the scene with their location, 
        rotation (in degrees), and dimensions (width, depth, height in meters). 
        Use the 'dimensions' to perform clash detection and ensure new 
        components stay within the FIA Reference Volumes.
        """
        response = client.send_command("get_scene_info")
        if response.get("status") == "success":
            return str(response.get("result", []))
        return f"Error getting scene info: {response.get('message')}"

    # Return both tools
    return [execute_blender_visualization, get_scene_info]


class Backend():
    def __init__(self, model_name, milvus_host, milvus_port, debug=False, fresh=False, blender_mcp=False):
        self.model_name = model_name
        self.debug = debug
        self.blender_mcp = blender_mcp
        self.mtype = _model_type(model_name)

        # --- LLM setup ---
        if self.mtype == "deepseek":
            self.llm = ChatOpenAI(
                openai_api_base=VLLM_API_BASE,
                openai_api_key="unused",
                model_name=model_name
            )
        elif self.mtype == "o3":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=1,  # o3 only accepts temperature=1
            )
        else:
            self.llm = ChatOpenAI(model=model_name)


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
            search_kwargs={"k": 5}
        )

        # Build the retrieval chain (used when blender_mcp is off)
        rag_prompt = _build_rag_chat_prompt(model_name)
        combine_docs_chain = create_stuff_documents_chain(self.llm, rag_prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

        # Set up the Blender agent (used when blender_mcp is on)
        if blender_mcp:
            from langchain.agents import AgentExecutor

            blender_tools = _setup_blender_tools()
            self.tools = blender_tools

            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])

            if self.mtype == "o3":
                from langchain.agents import create_openai_tools_agent
                agent = create_openai_tools_agent(self.llm, self.tools, agent_prompt)
            else:
                from langchain.agents import create_openai_functions_agent
                agent = create_openai_functions_agent(self.llm, self.tools, agent_prompt)

            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True
            )

        # user needs to upload document
        self.documents_uploaded = True


    def inference(self, user_text):
        if not self.documents_uploaded:
            return "Please upload a document."

        thought_content = None
        response = None

        if self.blender_mcp:
            # Agent path: retrieve context manually, let agent decide on tools
            docs = self.retriever.get_relevant_documents(user_text)
            context = "\n".join([doc.page_content for doc in docs])

            if self.debug:
                self._log_docs(docs)

            result = self.agent_executor.invoke({"input": user_text, "context": context})
            response = result["output"]

        elif self.mtype == "o3":
            # o3 path: manual retrieval + direct LLM call to preserve AIMessage
            docs = self.retriever.get_relevant_documents(user_text)
            context = "\n".join([doc.page_content for doc in docs])

            if self.debug:
                self._log_docs(docs)

            system_text = SIMPLE_RAG_PROMPT.format(context=context)
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=user_text)
            ]
            ai_message = self.llm.invoke(messages)
            response = ai_message.content

            # Extract o3 reasoning from the AIMessage
            reasoning = getattr(ai_message, "additional_kwargs", {}).get("reasoning_content")
            if not reasoning:
                reasoning = getattr(ai_message, "response_metadata", {}).get("reasoning_content")
            if reasoning:
                thought_content = reasoning

        else:
            # Standard / DeepSeek path: use retrieval chain
            chain_response = self.qa_chain.invoke({"input": user_text})
            response = chain_response["answer"]

            if self.debug:
                self._log_retrieved_context(chain_response)

        # --- Yield thought + answer ---
        # o3 thinking (from reasoning_content field)
        if thought_content:
            yield {"type": "thought_start", "content": thought_content}
            yield {"type": "thought_end", "content": thought_content}

        # DeepSeek-R1 thinking (from <think> tags in text)
        if self.mtype == "deepseek" and "</think>" in response:
            if "<think>" in response:
                before_think, remainder = response.split("<think>", 1)
                if before_think.strip():
                    yield {"type": "answer", "content": before_think}
                think_text, answer = remainder.split("</think>", 1)
            else:
                think_text, answer = response.split("</think>", 1)

            yield {"type": "thought_start", "content": think_text.strip()}
            yield {"type": "thought_end", "content": think_text.strip()}

            if answer.strip():
                yield {"type": "answer", "content": answer.strip()}
        else:
            yield {"type": "answer", "content": response}


    def _log_docs(self, docs):
        """Log retrieved documents to terminal."""
        print("\n" + "=" * 80)
        print(f"[DEBUG] Retrieved {len(docs)} context document(s)")
        print("=" * 80)
        for i, doc in enumerate(docs):
            print(f"\n--- Document {i + 1} ---")
            print(f"  {doc.page_content}")
        print("=" * 80 + "\n")


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

