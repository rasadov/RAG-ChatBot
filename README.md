# RAG-ChatBot

A Retrieval-Augmented Generation (RAG) based chatbot that combines the power of GPT-4 with custom knowledge retrieval capabilities. The chatbot can process and answer questions based on provided URLs and generate quizzes from the content.

## Features

- **RAG Implementation**: Combines vector embeddings with GPT-4 for context-aware responses
- **Custom Knowledge Base**: Build vector store from provided URLs
- **Quiz Generation**: Generate quizzes in Russian from the knowledge base
- **Smart Text Chunking**: Intelligent text segmentation for optimal context retrieval
- **Vector Caching**: Local caching of vector embeddings for improved performance

## Prerequisites

- Python 3.8+
- OpenAI API key
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAG-ChatBot.git
cd RAG-ChatBot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_api_key_here
EMBEDDING_MODEL=text-embedding-ada-002  # or your preferred model
```

## Usage

### Basic Usage

```python
from src.chatbot.service import ChatBotService

# Initialize the chatbot with your URLs
urls = ["https://example.com/page1", "https://example.com/page2"]
chatbot = ChatBotService(urls)

# Get a response
response = chatbot.get_response("What is the main topic?")
print(response)

# Generate a quiz
quiz = chatbot.generate_test("specific topic")
print(quiz)
```

### Quiz Generation

The chatbot can generate quizzes in Russian from the provided content:

```python
# Generate a quiz on a specific topic
quiz = chatbot.generate_test("artificial intelligence")

# Generate a random quiz
quiz = chatbot.generate_test()
```

## How It Works

1. **Text Processing**:
   - URLs are scraped for content
   - Text is chunked into manageable pieces
   - Chunks are converted to vector embeddings

2. **Query Processing**:
   - User query is converted to vector embedding
   - Similar chunks are retrieved using cosine similarity
   - Retrieved context is sent to GPT-4 for response generation

3. **Quiz Generation**:
   - Content is retrieved based on topic
   - GPT-4 generates structured quiz questions
   - Questions are formatted and presented to the user

## Testing

Run the test suite:
```bash
python -m unittest tests/test_chatbot.py
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT-4 and embedding APIs
- Contributors and maintainers of the project
