import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.getenv("SLACK_BOT_TOKEN")

from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool

tavily_tool = TavilySearchResults(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()

from typing import Annotated, List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )
    
    # Import required libraries and set up temporary directory for file operations
import yahoofinance as yf
from pathlib import Path
from tempfile import TemporaryDirectory

_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

# Import required tools and date handling
from langchain_core.tools import tool
from datetime import date

# Tool to get company information
@tool
def company_information(ticker: str) -> dict:
    """Use this tool to retrieve company information like address, industry, sector, company officers, business summary, website,
       marketCap, current price, ebitda, total debt, total revenue, debt-to-equity, etc."""

    ticker_obj = yf.Ticker(ticker)
    ticker_info = ticker_obj.get_info()

    return ticker_info

# Tool to get dividend and earnings dates
@tool
def last_dividend_and_earnings_date(ticker: str) -> dict:
    """
    Use this tool to retrieve company's last dividend date and earnings release dates.
    It does not provide information about historical dividend yields.
    """
    ticker_obj = yf.Ticker(ticker)

    return ticker_obj.get_calendar()

# Tool to get mutual fund holders information
@tool
def summary_of_mutual_fund_holders(ticker: str) -> dict:
    """
    Use this tool to retrieve company's top mutual fund holders.
    It also returns their percentage of share, stock count and value of holdings.
    """
    ticker_obj = yf.Ticker(ticker)
    mf_holders = ticker_obj.get_mutualfund_holders()

    return mf_holders.to_dict(orient="records")

# Tool to get institutional holders information
@tool
def summary_of_institutional_holders(ticker: str) -> dict:
    """
    Use this tool to retrieve company's top institutional holders.
    It also returns their percentage of share, stock count and value of holdings.
    """
    ticker_obj = yf.Ticker(ticker)
    inst_holders = ticker_obj.get_institutional_holders()

    return inst_holders.to_dict(orient="records")

# Tool to get stock grade changes
@tool
def stock_grade_upgrades_downgrades(ticker: str) -> dict:
    """
    Use this to retrieve grade ratings upgrades and downgrades details of particular stock.
    It'll provide name of firms along with 'To Grade' and 'From Grade' details. Grade date is also provided.
    """
    ticker_obj = yf.Ticker(ticker)

    curr_year = date.today().year

    upgrades_downgrades = ticker_obj.get_upgrades_downgrades()
    upgrades_downgrades = upgrades_downgrades.loc[upgrades_downgrades.index > f"{curr_year}-01-01"]
    upgrades_downgrades = upgrades_downgrades[upgrades_downgrades["Action"].isin(["up", "down"])]

    return upgrades_downgrades.to_dict(orient="records")

# Tool to get stock split history
@tool
def stock_splits_history(ticker: str) -> dict:
    """
    Use this tool to retrieve company's historical stock splits data.
    """
    ticker_obj = yf.Ticker(ticker)
    hist_splits = ticker_obj.get_splits()

    return hist_splits.to_dict()

# Tool to get stock news
@tool
def stock_news(ticker: str) -> dict:
    """
    Use this to retrieve latest news articles discussing particular stock ticker.
    """
    ticker_obj = yf.Ticker(ticker)

    return ticker_obj.get_news()

tools = [
         company_information,
         last_dividend_and_earnings_date,
         stock_splits_history,
         summary_of_mutual_fund_holders,
         summary_of_institutional_holders,
         stock_grade_upgrades_downgrades,
         stock_news
]

from langchain_core.messages import HumanMessage

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import Literal

# Define the members and system prompt
members = ["Researcher", "Coder", "Chart_generator"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

# Define the options for the next step
options = ["FINISH"] + members

# Correct usage of Literal without unpacking
class RouteResponse(BaseModel):
    next: Literal['FINISH', 'Researcher', 'Coder','Chart_generator']  # Pass options directly

# Create a prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-4-0125-preview")

# Define the supervisor agent function
def supervisor_agent(state):
    supervisor_chain = (
        prompt
        | llm.with_structured_output(RouteResponse)
    )
    return supervisor_chain.invoke(state)

# Example usage of supervisor_agent function
# Example state can be defined here based on your use case
# state = {"messages": ... }
# response = supervisor_agent(state)
# print(response)

import functools
import operator
from typing import Sequence, TypedDict, Annotated
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image, display

# Add custom fonts
font_files = [
    './Prompt/Prompt-Regular.ttf',
    './Prompt/Prompt-Medium.ttf',
    './Prompt/Prompt-Bold.ttf',
    './Unbounded/static/Unbounded-Regular.ttf',
    './Unbounded/static/Unbounded-Bold.ttf'
]

# Register fonts with matplotlib
for font_path in font_files:
    fm.fontManager.addfont(font_path)

# Update matplotlib's font settings and style
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'Prompt'
plt.rcParams['font.weight'] = 'regular'
plt.rcParams['figure.facecolor'] = '#171717'
plt.rcParams['axes.facecolor'] = '#171717'
plt.rcParams['savefig.facecolor'] = '#171717'

# Load and display logo with consistent dark background
logo = mpimg.imread('navbarLogo (1).png')
fig_logo = plt.figure(figsize=(8, 4))
fig_logo.patch.set_facecolor('#171717')
plt.imshow(logo)

# Add logo as an inset
ax = plt.gca()
logo_ax = ax.inset_axes([0.85, 0.85, 0.15, 0.15])  # [x, y, width, height] in axis coordinates
logo_ax.imshow(logo)
logo_ax.axis('off')
logo_ax.patch.set_alpha(0.8)

plt.axis('off')
plt.show()

# Also display using IPython for simpler visualization
display(Image('navbarLogo (1).png'))

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent

# Define the base class for the agent's state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    company_name: str = "ai alpha"
    company_logo: str = "navbarLogo (1).png"
    company_colors: list = [
        '#F9B208',  # Gold
        '#F98404',  # Orange
        '#FC5404',  # Bright orange
        '#FFFFFF',  # Pure white
        '#FAFAFA',  # Almost white
        '#F5F5F5',  # Very light gray
        '#E5E5E5',  # Light gray
        '#D4D4D4',  # Medium light gray
        '#A3A3A3',  # Medium gray
        '#737373',  # Medium dark gray
        '#525252',  # Dark gray
        '#404040',  # Very dark gray
        '#262626',  # Almost black
        '#171717'   # Deep black
    ]
    company_fonts: list = [
        'Unbounded',  # For headlines
        'Prompt'      # For body text
    ]

# Create the research agent node
research_agent = create_react_agent(llm, tools=[tavily_tool])
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

# Create chart agent with enhanced aesthetic instructions
chart_system_prompt = """You are a chart generator that creates highly aesthetic and organized visualizations with ai alpha branding.
Follow these precise design specifications:

Background:
- Always use #171717 (deep black) as the consistent background color for ALL charts and visualizations
- Ensure background color is set for both figure and axes: 
  plt.rcParams['figure.facecolor'] = '#171717'
  plt.rcParams['axes.facecolor'] = '#171717'

Colors:
- Primary: #F9B208 (key highlights and main data)
- Secondary: #F98404 (supporting elements)
- Accent: #FC5404 (emphasis)
- Text: #FFFFFF (pure white)
- Grayscale Elements: Use predefined grays for subtle variations

Typography & Layout:
- Headlines: Unbounded font (24px)
- Body Text: Prompt font (14-18px)
- Labels: Prompt font (12-14px)
- Maintain consistent line spacing (1.5)
- Use 24px padding around elements

Logo Implementation (REQUIRED FOR ALL CHARTS):
- ALWAYS add this code after creating any chart:
  ax = plt.gca()
  logo = mpimg.imread('navbarLogo (1).png')
  logo_ax = ax.inset_axes([0.85, 0.85, 0.15, 0.15])
  logo_ax.imshow(logo)
  logo_ax.axis('off')
  logo_ax.patch.set_alpha(0.8)

Chart Guidelines:
- Set figure size BEFORE plotting: plt.figure(figsize=(16, 9))
- Use plt.tight_layout() to prevent overlapping
- Clear axes ticks and labels when not needed
- Ensure high contrast between elements
- Include legends with clear color indicators
- Maintain clean grid lines at 15% opacity
- ALWAYS add the logo using the code above before showing or saving any chart"""

# Create the coder agent node
code_agent = create_react_agent(llm, tools=[python_repl_tool], state_modifier=chart_system_prompt)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

chart_agent = create_react_agent(
    llm, 
    tools=[python_repl_tool],
    state_modifier=chart_system_prompt
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="Chart_generator")

search_agent = create_react_agent(llm, tools=tools, state_modifier="You are a data gatherer. Your role is to collect comprehensive financial and market data using available tools. Always ensure you gather complete datasets including historical prices, trading volumes, and relevant financial metrics that will be essential for creating detailed visualizations.")
search_node = functools.partial(agent_node, agent=search_agent, name="Search")

webscraper_agent = create_react_agent(llm, tools=[scrape_webpages], state_modifier="You are a web research specialist. Focus on gathering contextual information, news, and analysis that can complement the financial data. Look for market trends, company announcements, and analyst reports that could provide valuable context for the visualization.")
webscraper_node = functools.partial(agent_node, agent=webscraper_agent, name="WebScraper")

# Define supervisor_chain properly using prompt and llm as per your setup
supervisor_chain = (
    prompt
    | llm.with_structured_output(RouteResponse)
)

# Create the workflow graph and add nodes
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("Chart_generator", chart_node)
workflow.add_node("supervisor", supervisor_chain)

# First, clear any existing workflow
workflow = StateGraph(AgentState)

# Define members and their nodes
members = ["Researcher", "Coder", "Chart_generator", "WebScraper", "Search"]
nodes = {
    "Researcher": research_node,
    "Coder": code_node,
    "Chart_generator": chart_node,
    "WebScraper": webscraper_node,
    "Search": search_node,
    "supervisor": supervisor_chain
}

# Add all nodes to the workflow
for name, node in nodes.items():
    workflow.add_node(name, node)

# Define the routing function
def route_next(state):
    next_step = state["next"]
    if next_step in members:
        return next_step
    elif next_step == "FINISH":  # Changed from "FINISH" to "__end__"
        return END
    else:
        raise ValueError(f"Invalid next step: {next_step}")

# Add single conditional edge from supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_next,
    {**{member: member for member in members}, "__end__": END}  # Changed from "FINISH" to "__end__"
)

# Add edges from workers back to supervisor
for member in members:
    workflow.add_edge(member, "supervisor")

# Add entrypoint
workflow.add_edge(START, "supervisor")

# Compile the graph
graph = workflow.compile()

def enter_chain(message: str):
    # Add a fixed branding extraction prompt
    results = {
        "messages": [
            HumanMessage(content=message)
        ],
    }
    return results


research_chain = enter_chain | graph