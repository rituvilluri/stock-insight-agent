import chainlit as cl
from langchain_core.messages import HumanMessage
import json
import plotly.io as pio

# Import the compiled workflow from the new structure
from agent.graph.workflow import app

@cl.on_chat_start
async def start():
    # Create a beautiful welcome message
    welcome_content = """🚀 **Welcome to the Stock Insight Agent!** 📊

I'm your AI-powered assistant for stock market analysis. Here's what I can help you with:

💡 **Ask me about:**
• "How did NVIDIA perform over the last 3 weeks?"
• "What's Apple's stock performance for the past month?"
• "Show me Tesla's stock data from last week"
• "Generate a chart for Microsoft stock"

🎯 **I can:**
• Parse natural language date ranges
• Fetch historical stock data
• Generate interactive stock charts
• Provide detailed performance analysis

Ready to analyze some stocks? Just ask! 📈"""
    
    await cl.Message(
        content=welcome_content,
        author="Stock Insight Agent"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Show typing indicator
    await cl.Message(
        content="Analyzing your request...",
        author="Stock Insight Agent"
    ).send()
    
    state = {
        "messages": [HumanMessage(content=message.content)],
        "next": "call_tool"
    }
    
    try:
        result = app.invoke(state)
        last_message = result["messages"][-1]
        
        content = last_message.content

        # Handle inline Plotly JSON payload from the chart tool
        if isinstance(content, str) and content.startswith("PLOTLY_JSON:"):
            fig_json = content[len("PLOTLY_JSON:"):]
            fig = pio.from_json(fig_json)

            await cl.Message(
                content="📊 **Here's your stock chart:**",
                author="Stock Insight Agent",
                elements=[cl.Plotly(name="Stock Chart", figure=fig)]
            ).send()

        elif isinstance(content, str) and content.endswith('.html'):
            # Fallback: raw HTML path (shown as text if HTML rendering disabled)
            await cl.Message(
                content="📊 **Here's your stock chart:**",
                author="Stock Insight Agent"
            ).send()

            await cl.Message(
                content=content,
                author="Stock Insight Agent"
            ).send()

        else:
            # For regular responses, format nicely
            formatted_content = f"📈 **Analysis Complete!**\n\n{content}"
            
            await cl.Message(
                content=formatted_content,
                author="Stock Insight Agent"
            ).send()
            
    except Exception as e:
        error_msg = f"""
        ❌ **Oops! Something went wrong:**
        
        {str(e)}
        
        Please try rephrasing your question or ask about a different stock.
        """
        await cl.Message(
            content=error_msg,
            author="Stock Insight Agent"
        ).send()