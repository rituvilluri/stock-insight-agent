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
6. [x] Interactive stock chart generation
7. [x] LangGraph workflow refactoring
8. [ ] RAG implementation
9. [ ] News analysis
10. [ ] Social media integration
11. [ ] Options analysis

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
│   └── graph/               # LangGraph workflow implementation
│       ├── nodes/
│       │   └── tool_caller.py    # Main tool execution node
│       ├── edges/
│       │   └── decision_router.py # Workflow decision logic
│       └── workflow.py           # Compiled LangGraph workflow
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
│   └── llm_setup.py         # LLM configuration (Groq Llama 3.1)
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
3. [x] Refactor LangGraph workflow into proper nodes/edges structure
4. [x] Implement interactive stock chart generation
5. [ ] Implement RAG system for enhanced analysis
6. [ ] Add news analysis capabilities
7. [ ] Develop social media sentiment analysis
8. [ ] Add options data analysis

## 🎨 Recent Implementation Updates

### Interactive Chart Generation (Latest)
**Implementation Date**: September 2025
**Key Features**:
- Interactive Plotly candlestick charts
- Inline chart rendering in Chainlit
- Support for multiple time periods (1mo, 3mo, 1y)
- Automatic chart generation from natural language requests

**Technical Implementation**:
- Added `get_stock_chart` function to `agent/graph/nodes/tool_caller.py`
- Integrated Plotly JSON serialization for Chainlit compatibility
- Updated Chainlit app to handle `cl.Plotly` elements
- Chart storage in `app/chainlit/static/charts/` directory

**Usage Examples**:
```
User: "Generate a chart for NVDA for the last 3 months"
→ Returns interactive candlestick chart with price data
```

### LangGraph Workflow Refactoring
**Implementation Date**: September 2025
**Key Changes**:
- Separated workflow into proper nodes and edges structure
- Created `agent/graph/nodes/tool_caller.py` for tool execution
- Created `agent/graph/edges/decision_router.py` for workflow routing
- Updated `agent/graph/workflow.py` to compile the complete workflow

**Benefits**:
- Better code organization and maintainability
- Easier to add new tools and workflow logic
- Cleaner separation of concerns
- More scalable architecture

### LLM Model Updates
**Implementation Date**: September 2025
**Changes**:
- Updated from `llama3-70b-8192` (decommissioned) to `llama-3.1-8b-instant`
- Maintained free Groq API usage
- Improved response quality and speed

## 📌 Important Reminders
- Always use free resources where possible
- Document all major decisions
- Regular commits for tracking progress
- Test each feature before moving to next
- Keep dependencies updated
- Maintain good code documentation
- Prefer simple, direct function calls over complex wrappers
- Use `PYTHONPATH=.` when running Chainlit to resolve module imports

## 🎯 Monetization Strategy & Real-Time Features

### Market Analysis
**Competition & Opportunity:**
- **Existing Players**: Finviz, Stocktwits, Seeking Alpha, Bloomberg Terminal
- **Market Gap**: No platform combines news + social + options + real-time monitoring
- **Pricing Gap**: Most solutions are expensive ($50-200/month) vs. Bloomberg ($2k/month)
- **Unique Value**: Holistic analysis + real-time monitoring + AI insights

### MVP Strategy (Phase 1 - Free)
**Focus on Historical Analysis:**
```
User Queries:
├── "What was Tesla sentiment last week?"
├── "Show me options activity for NVIDIA last month"
├── "News + social + options correlation analysis"
└── "How did sentiment correlate with stock price movement?"
```

**Technical Implementation:**
- RAG system for news articles
- Historical social media sentiment analysis
- Options data correlation
- Multi-source data aggregation

### Premium Features (Phase 2 - Paid)
**Real-Time Monitoring:**
```
Monitoring Commands:
├── "Monitor Tesla for 5 minutes" ($5/session)
├── "Watch NVIDIA sentiment for 1 hour" ($20/hour)
├── "Daily alerts for Apple" ($50/month)
└── "Real-time options flow monitoring" ($100/month)
```

## 🔄 Real-Time Monitoring Technical Architecture

### MCP (Model Context Protocol) Integration
**Why MCPs for Real-Time:**
- **Long-running connections** (1+ hour monitoring)
- **Streaming data feeds** (real-time sentiment/options)
- **Event-driven alerts** (threshold-based notifications)
- **Stateful monitoring** (remember what you're watching)

### Technical Implementation Plan

#### MCP Server Architecture:
```python
# mcp_server.py
class FinancialDataMCPServer:
    def __init__(self):
        self.twitter_stream = TwitterStream()
        self.options_feed = OptionsDataFeed()
        self.news_monitor = NewsMonitor()
        self.alert_engine = AlertEngine()
    
    async def start_monitoring(self, symbol, duration, thresholds):
        # Start real-time data collection
        # Monitor for threshold breaches
        # Send alerts via Chainlit
```

#### Data Sources (Free Tier Limitations):
- **Twitter API**: 500 requests/month (very limited)
- **Reddit API**: Free but rate-limited
- **Yahoo Finance**: Free but no real-time streaming
- **Alpha Vantage**: 5 API calls/minute (very limited)

#### Real-Time Monitoring Flow:
```
User: "Watch Tesla for 1 hour"
↓
MCP Server:
├── Twitter sentiment stream
├── Options data feed
├── News monitoring
└── Alert engine
↓
Real-time alerts:
├── "Tesla sentiment dropped 15%"
├── "Put volume increased 200%"
├── "Breaking news: Tesla recall"
└── "Recommendation: Consider closing call position"
```

### Implementation Phases

#### Phase 1: Historical Analysis (Free MVP)
**Tools to Build:**
1. **News RAG Tool**
   - Location: `tools/news/news_rag_tool.py`
   - Purpose: Retrieve relevant news articles
   - Data: Historical news articles
   - Output: Summarized news insights

2. **Social Sentiment Tool**
   - Location: `tools/social/sentiment_analyzer_tool.py`
   - Purpose: Analyze historical social media sentiment
   - Data: Past tweets, Reddit posts
   - Output: Sentiment scores and trends

3. **Options Analysis Tool**
   - Location: `tools/options/options_analyzer_tool.py`
   - Purpose: Historical options data analysis
   - Data: Past options activity
   - Output: Put/call ratios, volume analysis

#### Phase 2: Real-Time Monitoring (Premium)
**MCP Server Components:**
1. **Real-Time Data Collection**
   - Twitter sentiment streaming
   - Live options data feeds
   - Breaking news alerts

2. **Alert System**
   - Threshold-based notifications
   - Pattern recognition
   - Risk assessment

3. **User Interface**
   - Real-time monitoring dashboard
   - Alert management
   - Position tracking

### Technical Challenges & Solutions

#### Rate Limiting Issues:
**Problem**: Free APIs have very low limits
**Solutions**:
- Start with 5-minute monitoring sessions
- Implement smart caching
- Use multiple free data sources
- Gradual upgrade to paid APIs

#### Data Quality:
**Problem**: Noise in social media data
**Solutions**:
- Sentiment filtering algorithms
- Volume-based relevance scoring
- Multi-source verification

#### Cost Management:
**Problem**: Real-time APIs are expensive
**Solutions**:
- Freemium model (free historical, paid real-time)
- Usage-based pricing
- Tiered subscription plans

### Revenue Model
```
Free Tier:
├── Historical analysis (last 30 days)
├── Basic sentiment analysis
└── Limited news articles

Premium Tiers:
├── Basic ($20/month): 5-minute monitoring sessions
├── Pro ($50/month): 1-hour monitoring + daily alerts
└── Enterprise ($200/month): Unlimited monitoring + API access
```

### Success Metrics
- **User Engagement**: Daily active users
- **Feature Usage**: Monitoring session frequency
- **Revenue**: Monthly recurring revenue (MRR)
- **User Retention**: Churn rate
- **Market Validation**: User feedback and feature requests 