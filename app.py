import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool, TXTSearchTool, DOCXSearchTool, BrowserbaseLoadTool, CodeInterpreterTool
from langchain.llms import OpenAI  # or the specific class you need for OpenRouter

# Set environment variables using secrets stored in Streamlit secrets management
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

# Define LLM
llm = OpenAI(
    model="databricks/dbrx-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Initialize tools
pdf_tool = PDFSearchTool()
txt_tool = TXTSearchTool()
docx_tool = DOCXSearchTool()
browser_tool = BrowserbaseLoadTool()
code_tool = CodeInterpreterTool()

# Define Agents
# 1. Literature Search Agent
literature_search_agent = Agent(
    role='Literature Search Agent',
    goal='Search for academic papers and articles on specified topics using various search tools.',
    verbose=True,
    memory=True,
    llm=llm,
    backstory='A diligent researcher skilled at finding relevant literature in diverse databases and repositories.',
    tools=[pdf_tool, txt_tool, docx_tool, browser_tool]
)

# 2. Data Extraction Agent
data_extraction_agent = Agent(
    role='Data Extraction Agent',
    goal='Extract detailed information from retrieved articles, including full-text content and key metadata.',
    verbose=True,
    memory=True,
    llm=llm,
    backstory='An expert in parsing and organizing complex data, ensuring all necessary details are captured.',
    tools=[pdf_tool, txt_tool, docx_tool, code_tool]
)

# 3. Summarization Agent
summarization_agent = Agent(
    role='Summarization Agent',
    goal='Generate summaries of collected literature, highlighting key findings and relevance.',
    verbose=True,
    memory=True,
    llm=llm,
    backstory='An excellent summarizer capable of distilling complex research into concise, understandable summaries.',
    tools=[txt_tool]
)

# 4. Bibliography Agent
bibliography_agent = Agent(
    role='Bibliography Agent',
    goal='Compile and format references into a well-organized bibliography.',
    verbose=True,
    memory=True,
    llm=llm,
    backstory='A meticulous organizer with an eye for detail, ensuring accurate and well-formatted bibliographies.',
    tools=[code_tool]
)

# Define Tasks
# Task 1: Literature Search Task
literature_search_task = Task(
    description=(
        "Perform searches on various academic databases and repositories using specified keywords. "
        "Retrieve metadata such as title, authors, and abstract of the found articles."
    ),
    expected_output='A list of articles with metadata including title, authors, and abstracts.',
    agent=literature_search_agent,
    tools=[pdf_tool, browser_tool]
)

# Task 2: Data Extraction Task
data_extraction_task = Task(
    description=(
        "Extract detailed information from retrieved articles, including full-text content and key metadata. "
        "Parse and structure the data for easy analysis."
    ),
    expected_output='Structured data including full-text content, abstracts, and metadata.',
    agent=data_extraction_agent,
    tools=[pdf_tool, txt_tool, docx_tool, code_tool]
)

# Task 3: Summarization Task
summarization_task = Task(
    description=(
        "Generate concise summaries of each article, highlighting the main findings and relevance. "
        "Provide an overview of novel contributions and research gaps."
    ),
    expected_output='Summaries highlighting key findings and relevance of each article.',
    agent=summarization_agent,
    tools=[txt_tool]
)

# Task 4: Bibliography Compilation Task
bibliography_task = Task(
    description=(
        "Compile and format references from the extracted data. "
        "Organize references according to specified guidelines and export in various formats."
    ),
    expected_output='Formatted bibliography in APA, MLA, or other specified formats.',
    agent=bibliography_agent,
    tools=[code_tool]
)

# Forming the crew
literature_search_crew = Crew(
    agents=[literature_search_agent, data_extraction_agent, summarization_agent, bibliography_agent],
    tasks=[literature_search_task, data_extraction_task, summarization_task, bibliography_task],
    process=Process.sequential
)

# Streamlit app interface
st.title("Automated Literature Search Crew")

keywords = st.text_input("Enter keywords for the literature search (e.g., gene therapy, CRISPR, genetic disorders):")

if st.button("Search"):
    if keywords:
        st.write("Performing literature search...")
        result = literature_search_crew.kickoff(inputs={'keywords': keywords})
        st.write(result)
    else:
        st.error("Please enter some keywords to perform the search.")
