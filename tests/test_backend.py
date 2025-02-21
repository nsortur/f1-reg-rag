import pytest
from unittest.mock import patch, MagicMock

from src.f1_llm.backend.backend import Backend

@pytest.fixture
def mock_auto_tokenizer():
    with patch("src.f1_llm.backend.backend.AutoTokenizer.from_pretrained") as mock:
        # return_value is what the patched method returns
        mock.return_value = MagicMock(name="MockAutoTokenizer")
        yield mock

@pytest.fixture
def mock_auto_model():
    with patch("src.f1_llm.backend.backend.AutoModelForCausalLM.from_pretrained") as mock:
        mock.return_value = MagicMock(name="MockAutoModel")
        yield mock

@pytest.fixture
def mock_pipeline():
    with patch("src.f1_llm.backend.backend.pipeline") as mock:
        mock.return_value = MagicMock(name="MockPipeline")
        yield mock

@pytest.fixture
def mock_ollama_embeddings():
    with patch("src.f1_llm.backend.backend.OllamaEmbeddings") as mock:
        mock.return_value = MagicMock(name="MockOllamaEmbeddings")
        yield mock

@pytest.fixture
def mock_milvus():
    with patch("src.f1_llm.backend.backend.Milvus") as mock:
        mock_milvus_instance = MagicMock(name="MockMilvusInstance")
        # as_retriever() might return something else
        mock_milvus_instance.as_retriever.return_value = MagicMock(name="MockRetriever")
        mock.return_value = mock_milvus_instance
        yield mock

@pytest.fixture
def backend_instance(
    mock_auto_tokenizer,
    mock_auto_model,
    mock_pipeline,
    mock_ollama_embeddings,
    mock_milvus
):
    """
    Returns an instance of the Backend class with all external dependencies mocked.
    """
    return Backend(
        model_name="fake-model",
        milvus_uri="http://fake-milvus:19530"
    )

def test_backend_init(
    mock_auto_tokenizer,
    mock_auto_model,
    mock_pipeline,
    mock_ollama_embeddings,
    mock_milvus,
    backend_instance
):
    """
    Test that the Backend __init__ calls the right dependencies and sets up class properties.
    """
    # Check that from_pretrained was called for both tokenizer and model
    mock_auto_tokenizer.assert_called_once_with("fake-model")
    mock_auto_model.assert_called_once_with(
        "fake-model",
        device_map="auto",
        trust_remote_code=True,
    )
    # Check pipeline was called with the mocked model & tokenizer
    assert mock_pipeline.called, "pipeline() was not called"
    
    # Check that we have an LLM and a retriever
    assert hasattr(backend_instance, "llm"), "Backend has no attribute 'llm'"
    assert hasattr(backend_instance, "retriever"), "Backend has no attribute 'retriever'"

    # documents_uploaded should be True initially
    assert backend_instance.documents_uploaded is True

def test_inference_when_documents_not_uploaded(backend_instance):
    """
    Test inference() should return the 'Please upload a document.' message 
    if documents are not uploaded.
    """
    # Force documents_uploaded to False
    backend_instance.documents_uploaded = False
    
    user_text = "Hello?"
    result = backend_instance.inference(user_text)
    assert result == "Please upload a document.", "Expected prompt to upload document"

def test_inference_normal_flow(backend_instance):
    """
    Test inference flow when documents are uploaded. We expect
    a mocked result from the QA chain.
    """
    # documents_uploaded is True by default
    # Patch RetrievalQA.from_chain_type to return a mock chain
    with patch("src.f1_llm.backend.backend.RetrievalQA.from_chain_type") as mock_chain_type:
        mock_chain_instance = MagicMock()
        mock_chain_instance.run.return_value = "Mocked QA Response"
        mock_chain_type.return_value = mock_chain_instance
        
        user_text = "What is the capital of France?"
        result = backend_instance.inference(user_text)
        
        mock_chain_type.assert_called_once()
        mock_chain_instance.run.assert_called_once_with(user_text)
        assert result == "Mocked QA Response"

def test_add_documents(backend_instance):
    """
    Test that add_documents calls the correct methods (loader, text splitter, vector store).
    """
    # Patch PyPDFLoader, RecursiveCharacterTextSplitter, and vector_store.add_documents
    with patch("src.f1_llm.backend.backend.PyPDFLoader") as mock_loader, \
         patch("src.f1_llm.backend.backend.RecursiveCharacterTextSplitter") as mock_splitter:
        
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = ["doc1", "doc2"]
        mock_loader.return_value = mock_loader_instance
        
        mock_splitter_instance = MagicMock()
        mock_splitter_instance.split_documents.return_value = ["split_doc1", "split_doc2"]
        mock_splitter.return_value = mock_splitter_instance
        
        # Also patch the vector_store's add_documents:
        backend_instance.vector_store.add_documents = MagicMock(name="add_documents")
        
        # Now call add_documents
        backend_instance.add_documents("some_file.pdf")
        
        # Check if loader.load() was called
        mock_loader_instance.load.assert_called_once()
        
        # Check if text_splitter.split_documents was called
        mock_splitter_instance.split_documents.assert_called_once_with(["doc1", "doc2"])
        
        # Check if vector_store.add_documents() was called with correct splits
        backend_instance.vector_store.add_documents.assert_called_once_with(["split_doc1", "split_doc2"])
        
        # Finally, verify that documents_uploaded was set to True
        assert backend_instance.documents_uploaded is True
