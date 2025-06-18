# Stock Insight Agent 📈🧠

A LangChain-powered agent that analyzes historical stock prices, news sentiment, and key events (like earnings) to infer whether investors likely profited or lost.

## 🎯 Project Goals
- Analyze stock price movements in relation to market events
- Provide context about why stocks moved (earnings, news, social sentiment)
- Track options performance during specific events
- Help make informed investment decisions based on historical patterns

## 🔧 Current Features
- 📊 Retrieves historical stock prices for given time periods
- 📅 Natural language date parsing
- 🤖 Basic LangChain agent implementation
- 💬 Simple command-line interface

## 🚧 Planned Features
- 📰 News article analysis and sentiment
- 💬 Social media sentiment (Twitter, Reddit)
- 📈 Options price tracking and analysis
- 🎨 Streamlit frontend
- 🔍 RAG implementation with ChromaDB
- 🧠 Migration to LangGraph for better agent workflows

## 🏗️ Architecture
Current:
```
stock-insight-agent/
├── agent/
│   ├── agent_executor.py
│   └── tools_init.py
├── tools/
│   ├── stockprice/
│   └── date/
├── llm/
└── app.py
```

Planned:
```
stock-insight-agent/
├── app/
│   ├── streamlit_app.py
│   └── components/
├── agent/
│   ├── graph/
│   │   ├── nodes/
│   │   └── edges/
│   └── tools/
├── data/
│   ├── vector_store/
│   └── embeddings/
├── services/
│   ├── news/
│   ├── social/
│   └── market/
└── utils/
```

## 🚀 How to Run

```bash
# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🔑 Environment Variables
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
```

## 🛠️ Tech Stack
- Python
- LangChain (to be migrated to LangGraph)
- Groq (Llama 3 API)
- ChromaDB (planned)
- Streamlit (planned)

## 📝 Notes
- Using free resources where possible
- Built as a learning project
- Focus on understanding market movements and patterns

```bash
pip install -r requirements.txt
streamlit run app.py
