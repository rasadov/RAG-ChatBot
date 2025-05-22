import re

import telebot

from src.config import Settings
from src.chatbot.service import ChatBotService

BOT_TOKEN = Settings.BOT_TOKEN

bot = telebot.TeleBot(BOT_TOKEN)
chatbot_service = ChatBotService()
user_quiz_states = {}

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Hello! I am a chatbot. Ask me anything.")

@bot.message_handler(func=lambda message: True)
def echo_all(message):
    chat_id = message.chat.id
    query = message.text

    if chat_id in user_quiz_states and user_quiz_states[chat_id]["active"]:
        quiz_state = user_quiz_states[chat_id]
        current_index = quiz_state["current_index"]
        questions = quiz_state["questions"]

        correct_answer_line = questions[current_index]["answer"]
        
        match = re.search(r":\s*([ABCD])\b", correct_answer_line.upper())
        if match:
            correct_letter = match.group(1)
        else:
            correct_letter = "?"
        
        user_answer = re.sub(r"[()]", "", query.strip().upper())
        
        if user_answer == correct_letter:
            bot.reply_to(message, f"Правильно! {correct_answer_line}")
        else:
            bot.reply_to(message, f"Неправильно. {correct_answer_line}")

        explanation = questions[current_index].get("explanation", "")
        if explanation:
            bot.send_message(chat_id, explanation)

        quiz_state["current_index"] += 1
        if quiz_state["current_index"] < len(questions):
            next_q_num = quiz_state["current_index"] + 1
            next_q_text = chatbot_service.format_question(
                questions[quiz_state["current_index"]],
                question_number=next_q_num
            )
            bot.send_message(chat_id, next_q_text)
        else:
            bot.send_message(chat_id, "Test is finished. Goodbye!")
            quiz_state["active"] = False

    else:
        if query.startswith("/test"):
            topic = query[5:].strip()
            first_question = chatbot_service.generate_test(topic)
            
            user_quiz_states[chat_id] = {
                "active": True,
                "questions": chatbot_service.current_quiz["questions"],
                "current_index": 0
            }
            bot.reply_to(message, first_question)
        else:
            response = chatbot_service.get_response(query)
            bot.reply_to(message, response)
