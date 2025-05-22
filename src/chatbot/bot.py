import openai
import numpy as np

from src.config import Settings
from src.utils.caching import BaseVectorCacher, LocalVectorCacher
from src.utils.scraper import BS4Scraper

openai.api_key = Settings.OPENAI_API_KEY

class ChatBot:
    EMBEDDING_MODEL: str = Settings.EMBEDDING_MODEL
    vector_store: list = []
    cacher: BaseVectorCacher = LocalVectorCacher("vector_store.json")
    scraper: BS4Scraper = BS4Scraper()
    urls: list

    def get_embedding(self, text: str) -> list:
        """
        Returns the embedding vector for a given text using OpenAI embeddings.
        """
        try:
            result = openai.embeddings.create(
                model=self.EMBEDDING_MODEL,
                input=text
            )
            return result.data[0].embedding
        except Exception as e:
            print("Embedding error:", e)
            return []

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the cosine similarity between two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def chunk_text_by_paragraphs(text: str) -> list:
        """
        Divides text into chunks based on paragraphs.
        """
        text = text.replace("\r", "").strip()
        chunks = text.split("\n\n")
        return chunks

    @staticmethod
    def chunk_text_by_length(text: str, chunk_size=500, overlap=50) -> list:
        """
        Divides text into chunks based on a fixed character length.
        """
        text = text.replace("\n", " ").replace("\r", "").strip()
        tokens = text.split()
        
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_str = " ".join(chunk_tokens)
            chunks.append(chunk_str)
            start += (chunk_size - overlap)
        return chunks

    def build_vector_store(self, *args: str) -> None:
        """
        1. Load slides from JSON
        2. Fetch external text from a Coda URL
        3. Chunk everything
        4. Embed each chunk
        5. Store in memory as a list of dicts
        """
        self.vector_store == self.cacher.load_vector_store()
        if self.vector_store:
            return

        # Combine all text
        all_texts = [self.scraper.fetch_text_from_page(url) for url in args]

        # Function to process each document text
        for doc_text in all_texts:
            results = self.process_text(doc_text)
            self.vector_store.extend(results)

        self.cacher.cache_vector_store(self.vector_store)

    def process_text(self, doc_text):
            results = []
            if not doc_text.strip():
                return results

            chunks = ChatBot.chunk_text_by_paragraphs(doc_text)
            for chunk in chunks:
                emb = self.get_embedding(chunk)
                if emb:
                    results.append({
                        "text": chunk,
                        "embedding": emb
                    })
            return results

    def retrieve_top_chunks(self, query: str) -> list:
        """
        Embed the user query, compute cosine similarity vs. each chunk,
        and return the top_k chunks with highest similarity.
        """
        top_k=3
        query_emb = np.array(self.get_embedding(query))

        # Compute similarity for each chunk
        scored = []
        for entry in self.vector_store:
            entry_emb = np.array(entry["embedding"])
            sim = ChatBot.cosine_similarity(query_emb, entry_emb)
            scored.append((sim, entry["text"]))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top_k chunk texts
        return [text for _, text in scored[:top_k]]

    @staticmethod
    def get_gpt4_response(user_prompt: str, retrieved_chunks: list) -> str:
        """
        Pass user prompt plus the top retrieved chunks as context into GPT-4.
        """
        context_text = "\n\n".join(retrieved_chunks)
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a helpful assistant. Use the provided context to answer "
                    "the user’s question as accurately as possible."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\n---\nUser prompt: {user_prompt}"
            }
        ]

        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception as e:
            print("Error calling GPT-4:", e)
            return "Sorry, I ran into an error."
