from pymilvus import MilvusClient
from config import COLLECTION_NAME, MILVUS_URI
from embedding import EmbeddingService

class MilvusManager:
    def __init__(self):
        self.client = MilvusClient(uri=MILVUS_URI)
        self.collection_name = COLLECTION_NAME

    def create_collection_if_not_exsits(
        self,
        dim: int
    ):
        if self.client.has_collection(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=dim
        )

    def insert(self, docs):
        vectors = EmbeddingService.embed(docs)
        dim = len(vectors[0])
        self.create_collection_if_not_exsits(dim)
        data = [
            {
                "id": i,
                "vector": vectors[i],
                "text": docs[i],
                "subject": "history"
            }
            for i in range(len(docs))
        ]

        res = self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

        return res

    def search(self, question: str, limit: int = 3):
        query_vector = EmbeddingService.embed(question)[0]
        result = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=["text", "subject"]
        )

        contexts = []

        for item in result[0]:
            entity = item["entity"]

            contexts.append(entity["text"])

        return contexts