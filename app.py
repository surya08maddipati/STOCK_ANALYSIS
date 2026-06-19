# app.py
# Stock Analysis AI with CrewAI and Groq

import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
import time
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Stock Analysis AI",
    page_icon="📈",
    layout="wide"
)

# Create session with retry strategy
def get_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

st.title("📈 AI Stock Analysis System")
st.markdown("Multi-agent system powered by CrewAI and Groq")

# Sidebar for API keys
with st.sidebar:
    st.header("🔑 API Configuration")

    groq_key = st.text_input("Groq API Key (gsk_...)", type="password")
    st.markdown("[Get Groq Key](https://console.groq.com/keys)")

    serpapi_key = st.text_input("SerpAPI Key", type="password")
    st.markdown("[Get SerpAPI Key](https://serpapi.com/)")

    st.markdown("---")
    st.info("Groq is FREE with fast inference!")

# Check if keys are provided
if not groq_key or not serpapi_key:
    st.warning("Please enter your API keys in the sidebar to continue")
    st.stop()

# Set environment variables
os.environ["GROQ_API_KEY"] = groq_key
os.environ["SERPAPI_API_KEY"] = serpapi_key

# Initialize LLM
@st.cache_resource
def get_llm(_groq_key):
    return LLM(
        model="groq/llama-3.3-70b-versatile",
        api_key=_groq_key,
        temperature=0.7
    )

llm = get_llm(groq_key)

# Define Tools
class StockSearchInput(BaseModel):
    query: str = Field(..., description="Search query")

class YahooFinanceInput(BaseModel):
    ticker: str = Field(..., description="Stock ticker")

class StockSearchTool(BaseTool):
    name: str = "stock_news_searcher"
    description: str = "Search latest stock news by ticker or company name. Returns brief headlines and snippets."
    args_schema: Type[BaseModel] = StockSearchInput

    def _run(self, query: str) -> str:
        api_key = os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            return "SerpAPI key is missing."

        params = {
            "engine": "google",
            "q": query,
            "tbm": "nws",
            "num": 3,
            "api_key": api_key
        }
        try:
            session = get_session()
            response = session.get("https://serpapi.com/search", params=params, timeout=15)
            if response.status_code != 200:
                return f"SerpAPI error {response.status_code}: {response.text[:300]}"

            results = response.json()
            news = results.get("news_results", results.get("organic_results", []))

            if not news:
                return "No recent news found for this query."

            output = []
            for item in news[:3]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                if title:
                    output.append(f"• {title}: {snippet[:100]}")

            return "\n".join(output) if output else "No relevant news found."
        except Exception as e:
            return f"News search failed: {type(e).__name__}: {str(e)}"


class YahooFinanceTool(BaseTool):
    name: str = "yahoo_finance_fetcher"
    description: str = "Get stock price data for a given ticker symbol."
    args_schema: Type[BaseModel] = YahooFinanceInput

    def _run(self, ticker: str) -> str:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")

            if hist.empty:
                return "No data"

            latest = hist.tail(5)
            current = latest['Close'].iloc[-1]
            change = ((latest['Close'].iloc[-1] - latest['Close'].iloc[0]) / latest['Close'].iloc[0]) * 100

            return f"""Price: ${current:.2f}
Change (1mo): {change:.2f}%
Recent closes: {', '.join(f'${c:.2f}' for c in latest['Close'])}"""
        except Exception as e:
            return f"Error fetching stock data: {str(e)}"


# Initialize tools
search_tool = StockSearchTool()
finance_tool = YahooFinanceTool()

# Create Agents
@st.cache_resource
def get_agents(_llm):
    analyst = Agent(
        role='Stock Analyst',
        goal='Analyze stocks',
        backstory='Financial expert',
        verbose=False,
        llm=_llm,
        tools=[search_tool, finance_tool]
    )

    writer = Agent(
        role='Report Writer',
        goal='Write reports',
        backstory='Financial writer',
        verbose=False,
        llm=_llm
    )

    return analyst, writer

analyst, writer = get_agents(llm)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input("Enter Stock Ticker", value="AAPL", max_chars=10).upper()

with col2:
    analyze_btn = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

if analyze_btn:
    if not ticker:
        st.error("Please enter a stock ticker")
    else:
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                news_data = search_tool.run(query=f"{ticker} stock news")
                price_data = finance_tool.run(ticker=ticker)

                report_prompt = f"""Write investment report for {ticker}.
Include: Summary, News, Price Analysis, Outlook.
Keep under 300 words.

News:
{news_data}

Price Analysis:
{price_data}
"""

                report_task = Task(
                    description=report_prompt,
                    expected_output="Investment report",
                    agent=writer
                )

                crew = Crew(
                    agents=[writer],
                    tasks=[report_task],
                    verbose=False
                )

                result = crew.kickoff()
                time.sleep(1)  # Add small delay to prevent connection issues

                # Convert result to string and escape dollar signs
                result_text = str(result).replace('$', r'\$')

                # Display results
                st.success("Analysis Complete!")

                st.markdown("---")
                st.markdown(f"## 📊 Investment Report - {ticker}")
                st.markdown(result_text)

                # Download button
                st.download_button(
                    label="📥 Download Report",
                    data=str(result),
                    file_name=f"{ticker}_analysis.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Powered by CrewAI, Groq, and SerpAPI**")