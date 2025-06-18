# Stock Insight Agent - Project Notes

## 📚 Learning Resources
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Chainlit Documentation](https://docs.chainlit.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## 🏗️ Architecture Decisions

### Current Architecture
The project uses LangGraph with Chainlit for the frontend:
- `app/chainlit/app.py`: Chainlit chat interface with LangGraph workflow
- `agent/graph/`: Contains LangGraph nodes and edges (placeholder)
- `tools/`: Simplified tools for stock analysis
- `llm/`: LLM configuration and setup

### Architecture Benefits
- LangGraph provides structured agent workflows
- Better state management and tool integration
- Chainlit offers modern chat-based UI
- Simplified tool structure for better maintainability

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

### Current Tools (Simplified)
1. Stock Analyzer Tool
   - Location: `tools/stockprice/stock_analyzer.py`
   - Purpose: Fetch and analyze historical stock data
   - Features: Dual data source (yfinance primary, Alpha Vantage fallback)
   - Dependencies: yfinance, requests

2. Date Parser Tool
   - Location: `tools/date/date_parser_tool.py`
   - Purpose: Convert natural language dates to ISO format
   - Features: Market-aware date parsing
   - Dependencies: dateutil

### Tool Simplification Benefits
- Removed heavy LangChain Tool wrappers
- Direct function calls for better performance
- Cleaner error handling
- Easier testing and debugging

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
3. [x] LangGraph migration
4. [x] Chainlit frontend
5. [x] Tool simplification
6. [ ] RAG implementation
7. [ ] News analysis
8. [ ] Social media integration
9. [ ] Options analysis

## 💡 Technical Decisions

### Why LangGraph?
- Better for complex agent workflows
- More flexible state management
- Active development
- Better suited for our use case

### Why Chainlit?
- Modern chat-based interface
- Better for conversational AI
- Real-time updates
- Easy integration with LangGraph

### Why ChromaDB?
- Free and open-source
- Local deployment
- Good performance
- Easy integration with LangChain/LangGraph

## 🔍 Code Structure Details

### Current File Structure
```
stock-insight-agent/
├── agent/
│   └── graph/               # LangGraph nodes and edges (placeholder)
├── app/
│   └── chainlit/
│       ├── app.py           # Main Chainlit application
│       ├── chainlit.md      # App description
│       └── static/charts/   # Generated stock charts
├── tools/
│   ├── stockprice/
│   │   └── stock_analyzer.py    # Consolidated stock analysis tool
│   └── date/
│       └── date_parser_tool.py  # Simplified date parsing
├── llm/
│   └── llm_setup.py         # LLM configuration
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

### Tool Architecture
- **stock_analyzer.py**: Single file containing all stock data functionality
  - Dual data source strategy (yfinance + Alpha Vantage)
  - Flexible ticker handling
  - Formatted output for better UX
- **date_parser_tool.py**: Lightweight date parsing
  - Market-aware date calculations
  - Support for relative date ranges
  - Clean error handling

## 🚀 Next Steps
1. [x] Simplify and consolidate tools
2. [x] Update Chainlit app to use new tools
3. [ ] Implement RAG system for enhanced analysis
4. [ ] Add news analysis capabilities
5. [ ] Develop social media sentiment analysis
6. [ ] Add options data analysis

## 📌 Important Reminders
- Always use free resources where possible
- Document all major decisions
- Regular commits for tracking progress
- Test each feature before moving to next
- Keep dependencies updated
- Maintain good code documentation
- Prefer simple, direct function calls over complex wrappers 