from abc import ABC, abstractmethod
from datetime import datetime
import json


class BaseVectorCacher(ABC):
    @abstractmethod
    def cache_vector_store(self, vector_store: dict,) -> None: ...
    @abstractmethod
    def load_vector_store(self,) -> dict: ...


class LocalVectorCacher(BaseVectorCacher):
    def __init__(self, cache_file: str) -> None:
        self.cache_file = cache_file

    def cache_vector_store(self, vector_store: dict,) -> None:
        """
        Save the vector store to disk for later use.

        # TO DO: Implement a more robust caching strategy (e.g., Postgres or even better NoSQL database).
        """
        if not vector_store:
            return
        data = {
            "vector_store": vector_store,
            "timestamp": str(datetime.now())
        }
        with open("vector_store.json", "w") as f:
            json.dump(data, f)

    def load_vector_store(self) -> dict:
        """
        Load the vector store from disk.

        # TO DO: Implement a more robust caching strategy (e.g., Postgres or even better NoSQL database).
        """
        try:
            with open("vector_store.json", "r") as f:
                data = json.load(f)
                vector_store = data.get("vector_store")
                timestamp = datetime.fromisoformat(data.get("timestamp"))
                difference = datetime.now() - timestamp
                if not vector_store or difference.days > 7:
                    print("Stored data is missing or too old. Rebuilding vector store...")
                    return None
                return vector_store
        except FileNotFoundError:
            return None
