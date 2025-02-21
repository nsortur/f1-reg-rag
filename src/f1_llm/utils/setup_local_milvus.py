from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


# Connect to Milvus
connections.connect(
    alias="default",
    host="127.0.0.1",
    port="19530"
)

# used with deepseek ollama
dim = 3584
collection_name = "test_collection"

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields)
collection = Collection(name=collection_name, schema=schema)
