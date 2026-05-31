from http import HTTPStatus

from dashscope import TextEmbedding
import dashscope

from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL

dashscope.api_key = DASHSCOPE_API_KEY


class EmbeddingService:
    BATCH_SIZE = 10

    @staticmethod
    def embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = [
            t.strip()
            for t in texts
            if t and t.strip()
        ]

        all_embeddings = []
        for i in range (0, len(texts), EmbeddingService.BATCH_SIZE):
            batch = texts[
                i: i + EmbeddingService.BATCH_SIZE
            ]

            print(
                f"embedding batch: "
                f"{i} ~ {i + len(batch)}"
            )

            resp = TextEmbedding.call(
                model=EMBEDDING_MODEL,
                input=batch
            )

            # 关键：检查是否成功
            if resp.status_code != HTTPStatus.OK:

                print("Embedding 调用失败")
                print("status_code:", resp.status_code)
                print("code:", resp.code)
                print("message:", resp.message)

                raise Exception(
                    f"Embedding failed: {resp.message}"
                )

            embeddings = [
                item["embedding"]
                for item in resp.output["embeddings"]
            ]

            all_embeddings.extend(embeddings)

        return all_embeddings
