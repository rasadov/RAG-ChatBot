import json
from datetime import datetime
import random

import openai
import numpy as np
import requests
from bs4 import BeautifulSoup

from config import Settings


openai.api_key = Settings.OPENAI_API_KEY

class ChatBot:
    EMBEDDING_MODEL: str = Settings.EMBEDDING_MODEL
    vector_store: list = []
    urls: list

    @staticmethod
    def fetch_text_from_page(url: str) -> str:
        """
        Example function to fetch text from a Coda link or any webpage.
        Adjust parsing depending on the HTML structure.
        """
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve content from {url}")
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all(["p", "span", "div"])
        
        # Join all text
        text = "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)
        return text.strip()

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
    def chunk_text(text: str, chunk_size=500, overlap=50) -> list:
        """
        Splits text into chunks of `chunk_size` characters with `overlap`.
        Adjust chunk_size/overlap to taste or do more advanced chunking (by sentences, etc.).
        """
        # Remove excessive whitespace, line breaks
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
        if self.load_vector_store():
            return

        # Combine all text
        all_texts = [ChatBot.fetch_text_from_page(url) for url in args]

        # Function to process each document text
        for doc_text in all_texts:
            results = self.process_text(doc_text)
            self.vector_store.extend(results)

        self.cache_vector_store()

    def process_text(self, doc_text):
            results = []
            if not doc_text.strip():
                return results

            chunks = ChatBot.chunk_text(doc_text, chunk_size=200, overlap=20)
            for chunk in chunks:
                emb = self.get_embedding(chunk)
                if emb:
                    results.append({
                        "text": chunk,
                        "embedding": emb
                    })
            return results

    def cache_vector_store(self) -> None:
        """
        Save the vector store to disk for later use.

        # TO DO: Implement a more robust caching strategy (e.g., Postgres or even better NoSQL database).
        """
        if not self.vector_store:
            return
        data = {
            "vector_store": self.vector_store,
            "timestamp": str(datetime.now())
        }
        with open("vector_store.json", "w") as f:
            json.dump(data, f)

    def load_vector_store(self) -> bool:
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
                    return False
                self.vector_store = vector_store
                return True
        except FileNotFoundError:
            return False

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

class ChatBotService:
    chatbot: ChatBot = ChatBot()
    current_quiz: dict

    def __init__(self, urls: list = []):
        self.current_quiz = None
        self.chatbot.urls = [] # Add your URLs here

        print("Building vector store. This may take some time...")
        self.chatbot.build_vector_store(*urls)
        
        print(f"Vector store built with {len(self.chatbot.vector_store)} chunks.")

    def get_response(self, user_input: str) -> str:
        if user_input.startswith("/test"):
            topic = user_input[5:].strip()
            return self.generate_test(topic)
        relevant_chunks = self.chatbot.retrieve_top_chunks(user_input)
        return ChatBot.get_gpt4_response(user_input, relevant_chunks)

    def generate_test(self, topic: str = "", num_questions: int = 5) -> str:
        """
        Generates a quiz in Russian, ensuring GPT doesn't embed answers inline.
        Stores the parsed quiz in self.current_quiz, returning the first question.
        """
        if not topic.strip():
            all_chunks = [entry["text"] for entry in self.chatbot.vector_store]
            random.shuffle(all_chunks)
            retrieved_chunks = all_chunks[:5]
        else:
            retrieved_chunks = self.chatbot.retrieve_top_chunks(topic, top_k=5)

        prompt_ru = (
            f"Составь викторину (примерно {num_questions} вопросов) по следующему контексту (строго на русском языке). "
            "ВАЖНО:\n"
            "1) Не указывай правильный ответ внутри строки с вопросом.\n"
            "2) Используй формат:\n"
            "   Вопрос 1: <сам вопрос>\n"
            "   A) ...\n"
            "   B) ...\n"
            "   C) ...\n"
            "   D) ...\n"
            "   Правильный ответ: ...\n"
            "   Пояснение: ...\n\n"
            "Контекст:\n\n"
        )

        context_text = "\n\n".join(retrieved_chunks)
        final_prompt = prompt_ru + context_text

        quiz_text = ChatBot.get_gpt4_response(final_prompt, retrieved_chunks)

        quiz_text = self.remove_inline_answers(quiz_text)

        quiz_questions = self.parse_quiz(quiz_text)

        self.current_quiz = {
            "questions": quiz_questions,
            "current_index": 0
        }

        if quiz_questions:
            return self.format_question(quiz_questions[0])
        else:
            return "Извините, не удалось сгенерировать тест."

    def remove_inline_answers(self, quiz_text: str) -> str:
        """
        Removes or separates occurrences of 'Ответ: X' if GPT still puts them in the same line
        as the question. This helps keep the final 'Правильный ответ: ...' lines clean.
        """
        import re
        pattern = r"(Вопрос\s*\d+:\s*.*?)(\s+Ответ\s*:\s*[ABCD])"
        cleaned_text = re.sub(pattern, r"\1", quiz_text, flags=re.IGNORECASE)
        return cleaned_text


    def parse_quiz(self, quiz_text: str) -> list:
        """
        Parses GPT output lines to form structured Q&A. 
        Looks for lines:
        - "Вопрос X:" or "1)" / "1."
        - "A)", "B)", "C)", "D)" for options
        - "Правильный ответ:" or "Correct answer:"
        - "Пояснение:" or "Explanation:"
        """
        import re

        lines = quiz_text.split("\n")
        questions = []
        current = {"question": "", "options": [], "answer": "", "explanation": ""}

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Treat "Вопрос X:" or "1)" / "1." as new question boundaries
            if (re.match(r"^Вопрос\s+\d+:", line_stripped, re.IGNORECASE)
                    or re.match(r"^\d+[\)\.]", line_stripped)):
                # If we have an existing question, append it
                if current["question"]:
                    questions.append(current)
                # Start a fresh one
                current = {"question": line_stripped, "options": [], "answer": "", "explanation": ""}

            elif re.match(r"^[ABCD][\)\.]", line_stripped, re.IGNORECASE):
                # It's an option line: A), B), etc.
                current["options"].append(line_stripped)

            elif (line_stripped.lower().startswith("правильный ответ:")
                or line_stripped.lower().startswith("correct answer:")):
                current["answer"] = line_stripped

            elif (line_stripped.lower().startswith("пояснение:")
                or line_stripped.lower().startswith("explanation:")):
                current["explanation"] = line_stripped

            else:
                if not current["answer"]:
                    current["question"] += " " + line_stripped
                else:
                    current["explanation"] += " " + line_stripped

        if current["question"]:
            questions.append(current)

        return questions

    def format_question(self, question_data: dict) -> str:
        """
        Return a nicely formatted string for Telegram:
        1) Question text
        2) A) ...
        3) B) ...
        ...
        """
        q_str = f"{question_data['question']}\n"
        for opt in question_data["options"]:
            q_str += f"{opt}\n"
        return q_str
