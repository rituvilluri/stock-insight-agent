# Stock Insight Agent - Project Notes

## 📚 Learning Resources
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## 🏗️ Architecture Decisions

### Current Architecture
The project currently uses a simple LangChain-based architecture:
- `app.py`: Entry point with CLI interface
- `agent/`: Contains agent logic and tool initialization
- `tools/`: Custom tools for stock analysis
- `llm/`: LLM configuration and setup

### Planned Architecture
Moving to LangGraph for better workflow management:
- More structured agent workflows
- Better state management
- Improved tool integration
- Enhanced reasoning capabilities

## 🔄 Git Workflow
Best practices for this project:
1. Commit after completing each feature
2. Use descriptive commit messages
3. Create branches for new features
4. Regular commits for documentation updates

### Suggested Commit Structure
```
feat: add new feature
docs: update documentation
refactor: restructure code
fix: bug fixes
test: add or update tests
```

## 📝 Implementation Notes

### Current Tools
1. Stock Price Tool
   - Location: `tools/stockprice/`
   - Purpose: Fetch historical stock data
   - Dependencies: Yahoo Finance API

2. Date Parser Tool
   - Location: `tools/date/`
   - Purpose: Convert natural language dates to ISO format
   - Dependencies: dateutil

### Planned Tools
1. News Analysis Tool
   - Purpose: Fetch and analyze news articles
   - Planned Dependencies: NewsAPI

2. Social Media Tool
   - Purpose: Analyze social media sentiment
   - Planned Dependencies: Reddit API, Twitter API

3. Options Analysis Tool
   - Purpose: Track options performance
   - Planned Dependencies: Yahoo Finance API

## 🎯 Project Milestones
1. [x] Basic stock price lookup
2. [x] Date parsing
3. [ ] LangGraph migration
4. [ ] RAG implementation
5. [ ] News analysis
6. [ ] Social media integration
7. [ ] Options analysis
8. [ ] Streamlit frontend

## 💡 Technical Decisions

### Why LangGraph?
- Better for complex agent workflows
- More flexible state management
- Active development
- Better suited for our use case

### Why ChromaDB?
- Free and open-source
- Local deployment
- Good performance
- Easy integration with LangChain/LangGraph

### Why Streamlit?
- Easy to learn
- Great for data visualization
- Real-time updates
- Good Python integration

## 🔍 Code Structure Details

### Current File Structure
```
stock-insight-agent/
├── agent/
│   ├── agent_executor.py      # Main agent logic
│   └── tools_init.py         # Tool initialization
├── tools/
│   ├── stockprice/
│   │   ├── stock_price_tool.py        # Tool definition
│   │   └── stock_price_retriever.py   # Data fetching logic
│   └── date/
│       └── date_parser_tool.py        # Date parsing logic
├── llm/
│   └── llm_setup.py          # LLM configuration
├── app.py                    # Main application
├── requirements.txt          # Dependencies
└── README.md                # Project documentation
```

### Empty Files Status
- `money_inference_tool.py`: Planned for financial sentiment analysis
- `option_price_tool.py`: Planned for options data analysis

## 🚀 Next Steps
1. Set up proper git repository
2. Create initial commit
3. Begin LangGraph migration
4. Implement RAG system
5. Add news analysis
6. Develop Streamlit frontend

## 📌 Important Reminders
- Always use free resources where possible
- Document all major decisions
- Regular commits for tracking progress
- Test each feature before moving to next
- Keep dependencies updated
- Maintain good code documentation 