import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Define LLM
llm = ChatOpenAI(
    model="databricks/dbrx-instruct",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Initialize the tool
search_tool = SerperDevTool(api_key=SERPER_API_KEY)

# Creating the agents
keyword_generator = Agent(
    role='Keyword Generator',
    goal='Generate search keywords using best practice guidelines based on the provided research title.',
    verbose=True,
    memory=True,
    backstory='You are skilled in creating effective search keywords to optimize literature search results.',
    llm=llm
)

google_scholar_searcher = Agent(
    role='Google Scholar Searcher',
    goal='Perform searches on Google Scholar using the generated keywords.',
    verbose=True,
    memory=True,
    backstory='You excel at finding relevant literature on Google Scholar using precise search queries.',
    tools=[search_tool],
    llm=llm
)

bibliographer = Agent(
    role='Bibliographer',
    goal='Compile a list of literature sources from the search results.',
    verbose=True,
    memory=True,
    backstory='You are meticulous and detail-oriented, ensuring that all relevant sources are compiled accurately.',
    llm=llm
)

# Creating the tasks
generate_keywords_task = Task(
    description='Use best practice guidelines to generate search keywords based on the provided research title. Research Title: {research_title}',
    expected_output='A list of search keywords.',
    agent=keyword_generator
)

perform_google_scholar_search_task = Task(
    description='Perform searches on Google Scholar using the generated keywords and collect relevant links. Keywords: {keywords}',
    expected_output='A list of relevant literature links.',
    tools=[search_tool],
    agent=google_scholar_searcher
)

compile_bibliography_task = Task(
    description='Compile all the relevant links found, creating a list of literature sources with titles and links. Links: {links}',
    expected_output='A comprehensive list of literature sources with titles and links.',
    agent=bibliographer
)

# Forming the crew
crew = Crew(
    agents=[keyword_generator, google_scholar_searcher, bibliographer],
    tasks=[generate_keywords_task, perform_google_scholar_search_task, compile_bibliography_task],
    process=Process.sequential
)

def kickoff(research_title):
    return crew.kickoff(inputs={'research_title': research_title})
