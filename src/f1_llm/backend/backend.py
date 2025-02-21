from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_milvus import Milvus
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI


class Backend():
    def __init__(self, model_name, milvus_host, milvus_port):

        if "DeepSeek-R1" in model_name:
            # load in the model from huggingface hub
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                            device_map="auto", 
                                                            trust_remote_code=True,)
            
            response_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer,
                                        max_new_tokens=128)
            self.llm = HuggingFacePipeline(pipeline=response_pipeline)
        
        else:
            self.llm = ChatOpenAI(name=model_name)


        # set up connection to RAG database
        embedding_function = OllamaEmbeddings(model='deepseek-r1')
        self.vector_store = Milvus(
            embedding_function=embedding_function,
            connection_args={"host": milvus_host, "port": milvus_port},
            collection_name="test_collection",
            auto_id=True,
            primary_field="id",
            vector_field="vector",
            text_field="text"
        )
        self.retriever = self.vector_store.as_retriever()

        # user needs to upload document
        self.documents_uploaded = True


    def inference(self, user_text):
        if not self.documents_uploaded:
            return "Please upload a document."

        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=self.retriever)
        response = qa_chain.run(user_text)
        return response


    def add_documents(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter()
        splits = text_splitter.split_documents(documents)

        # add splits using DeepSeek's embedding
        # independent of LLM, this is just used for context retrieval
        # TODO error handling
        _ = self.vector_store.add_documents(splits)
        self.documents_uploaded = True

