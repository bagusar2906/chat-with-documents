
from langchain.schema import Document
from langchain_community.vectorstores.supabase import SupabaseVectorStore
import numpy as np

# --- Custom Vector Store Class ---

class CustomeSupabaseVectorStore(SupabaseVectorStore):
    def __init__(self, client, embedding, table_name):
        self.client = client
        self.embedding = embedding
        self._embedding = embedding 
        self.table_name = table_name

    @classmethod
    def from_texts(cls, texts, embedding, client, table_name, metadata=None):
        vectors = embedding.embed_documents(texts)
        rows = []

        for i, text in enumerate(texts):
            row = {
                "source": f"{metadata[i]['source'] if metadata else 'unknown'}",
                "content": text,
                "embedding": vectors[i],
                "metadata": metadata[i] if metadata else {},
            }
            rows.append(row)

        insert_resp = client.table(table_name).insert(rows).execute()

        if insert_resp.data is None:
            raise Exception(f"Failed to insert documents")

        return cls(client=client, embedding=embedding, table_name=table_name)

    def similarity_search_by_vector_with_relevance_scores(
        self, embedding, k=4, filter=None, **kwargs
    ):
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        try:
            response = self.client.rpc(
                "match_documents",
                {
                    "query_embedding": embedding,
                    "match_count": k,
                    "filter": filter or {},
                },
            ).execute()
        except Exception as e:
            raise ValueError(f"Error querying Supabase: {e}")

        if not response.data:
            return []

        docs = []
        scores = []
        for match in response.data:
            metadata = match.get("metadata", {})
            text = match.get("content", "")
            score = match.get("similarity", 0)
            docs.append(Document(page_content=text, metadata=metadata))
            scores.append(score)

        return list(zip(docs, scores))