from pymilvus import MilvusClient, DataType
from app.config import COLLECTION_NAME, MILVUS_URI
from app.core.embedding import EmbeddingService

class MilvusManager:
    def __init__(self):
        self.client = MilvusClient(uri=MILVUS_URI)
        self.collection_name = COLLECTION_NAME

    def create_collection_if_not_exsits(
        self,
        dim: int
    ):
        # 总是删除旧 collection，确保使用新 schema
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
        
        # 定义明确的 schema
        schema = self.client.create_schema(auto_id=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="subject", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="metadata", datatype=DataType.JSON, max_length=65535)
        
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            consistency_level="Strong"
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

    def insert(self, docs, metadatas=None):
        vectors = EmbeddingService.embed(docs)
        dim = len(vectors[0])
        self.create_collection_if_not_exsits(dim)

        if metadatas is None:
            metadatas = [{} for _ in docs]

        data = []
        for vector, doc, metadata in zip(vectors, docs, metadatas):
            data.append({
                "vector": vector,
                "text": doc,
                "subject": metadata.get("subject", ""),
                "metadata": metadata,
            })

        result = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

        return result

    def search(self, question: str, limit: int = 3) -> list[dict]:
        self.client.load_collection(self.collection_name)
        query_vector = EmbeddingService.embed(question)[0]

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text", "metadata"],
        )

        if not results or not results[0]:
            return []

        return [
            {
                "text": hit["entity"]["text"],
                "page_num": hit["entity"].get("metadata", {}).get("page_num", 0),
                "distance": hit["distance"],
            }
            for hit in results[0]
        ]