import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.chatbot.bot import ChatBot
from src.chatbot.service import ChatBotService


class TestChatBot(unittest.TestCase):
    def setUp(self):
        self.chatbot = ChatBot()
        self.test_text = "This is a test text for embedding."
        self.test_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

    @patch('openai.embeddings.create')
    def test_get_embedding(self, mock_embedding):
        # Mock the OpenAI embedding response
        mock_embedding.return_value.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        
        result = self.chatbot.get_embedding(self.test_text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_cosine_similarity(self):
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        result = ChatBot.cosine_similarity(a, b)
        self.assertEqual(result, 0.0)  # Perpendicular vectors should have 0 similarity

    def test_chunk_text_by_paragraphs(self):
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        chunks = ChatBot.chunk_text_by_paragraphs(text)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Paragraph 1")
        self.assertEqual(chunks[1], "Paragraph 2")
        self.assertEqual(chunks[2], "Paragraph 3")

    def test_chunk_text_by_length(self):
        text = "This is a test text that should be chunked into smaller pieces"
        chunks = ChatBot.chunk_text_by_length(text, chunk_size=10, overlap=2)
        self.assertTrue(len(chunks) > 0)
        self.assertTrue(all(len(chunk.split()) <= 10 for chunk in chunks))

    @patch('openai.chat.completions.create')
    def test_get_gpt4_response(self, mock_gpt):
        mock_gpt.return_value.choices = [MagicMock(message=MagicMock(content="Test response"))]
        
        response = ChatBot.get_gpt4_response("test prompt", ["context1", "context2"])
        self.assertEqual(response, "Test response")


if __name__ == '__main__':
    unittest.main() 