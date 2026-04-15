# RAG AI Chatbot 🤖

A dockerized AI Python project of RAG chatbot. User can upload desired pdf documents on UI as the reference for the chatbot for answering questions. 

## 🧠 Technologies

- `Python`
- `Docker`
- `Open Router API`
- `ChromaDB`
- `LangChain`
- `Pypdf`
- `OpenAI`
- `Streamlit`

## ✏️ Installation & Preperation

1. Clone or download this repository

2. Set up environment variables:
    - `Copy the .env.example template file to create your .env file and add your API key:`
    - `OPENROUTER_API_KEY=your_openrouter_api_key_here`

3. Build and run the Docker container:
    - `Using Docker Compose:`
    - `docker-compose up --build`
    - `The app will be available at http://localhost:8501`
