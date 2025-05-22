import random

from chatbot import ChatBot

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
