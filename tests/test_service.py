import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.chatbot.bot import ChatBot
from src.chatbot.service import ChatBotService


class TestChatBotService(unittest.TestCase):
    def setUp(self):
        self.service = ChatBotService()
        self.test_quiz = {
            "questions": [
                {
                    "question": "Test question?",
                    "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
                    "answer": "Правильный ответ: A",
                    "explanation": "Test explanation"
                }
            ],
            "current_index": 0
        }

    def test_remove_inline_answers(self):
        quiz_text = "Вопрос 1: What is X? Ответ: A\nA) Option 1\nB) Option 2"
        cleaned = self.service.remove_inline_answers(quiz_text)
        self.assertNotIn("Ответ: A", cleaned)

    def test_parse_quiz(self):
        quiz_text = """
        Вопрос 1: Test question?
        A) Option 1
        B) Option 2
        C) Option 3
        D) Option 4
        Правильный ответ: A
        Пояснение: Test explanation
        """
        questions = self.service.parse_quiz(quiz_text)
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]["question"], "Вопрос 1: Test question?")
        self.assertEqual(len(questions[0]["options"]), 4)
        self.assertEqual(questions[0]["answer"], "Правильный ответ: A")
        self.assertEqual(questions[0]["explanation"], "Пояснение: Test explanation")

    def test_format_question(self):
        formatted = self.service.format_question(self.test_quiz["questions"][0])
        self.assertIn("Test question?", formatted)
        self.assertIn("A) Option 1", formatted)
        self.assertIn("B) Option 2", formatted)
        self.assertIn("C) Option 3", formatted)
        self.assertIn("D) Option 4", formatted)

    @patch('src.chatbot.service.ChatBot.get_gpt4_response')
    def test_get_response(self, mock_gpt):
        mock_gpt.return_value = "Test response"
        response = self.service.get_response("test query")
        self.assertEqual(response, "Test response")

if __name__ == '__main__':
    unittest.main() 