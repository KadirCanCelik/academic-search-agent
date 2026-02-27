import os
import sys
import re
import json
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.agents import AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.agents import AgentAction, AgentFinish
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.agents.format_scratchpad import format_log_to_str

from tools import search_arxiv, search_web, fetch_url_content, search_web_tavily, fetch_url_tavily

load_dotenv()

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
sec_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not sec_key:
    print("Token not found")
    exit()

try:
    stop_list = [
        "\nObservation", "Observation:", 
        "<|user|>", "<|assistant|>", 
        "\n\n\n", "Note:", 
        "(STOP HERE"
    ]

    llm_endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=512,
        do_sample=True,
        temperature=0.01, # Low temperature for factual consistency and precision
        top_k=50, #first 50
        top_p=0.95, # %95
        repetition_penalty=1.1,
        stop_sequences=stop_list,
        huggingfacehub_api_token=sec_key
    )

    llm = ChatHuggingFace(llm=llm_endpoint)

except Exception as e:
    print(f"Model Error: {e}")
    exit()


def safe_content_reader(input_text):
    """
    Validates the input string to ensure it is a proper URL.
    This acts as a guardrail to catch cases where the LLM might
    accidentally pass a title instead of a direct link.
    """
    # Normalize the input by removing potential whitespaces or surrounding quotes
    url = str(input_text).strip().strip('"').strip("'")

    # Ensure the string starts with a web protocol.
    if not url.startswith("http"):
        return f"ERROR: You provided a TITLE ('{url}'), but I need a valid URL starting with 'http'. Please check the 'link' field in the previous search results and copy that URL exactly."

    return fetch_url_content(url)

def safe_content_reader_tavily(input_text):
    """
    Validates the input string to ensure it is a proper URL.
    Uses Tavily extract for content fetching.
    """
    url = str(input_text).strip().strip('"').strip("'")

    if not url.startswith("http"):
        return f"ERROR: You provided a TITLE ('{url}'), but I need a valid URL starting with 'http'. Please check the 'link' field in the previous search results and copy that URL exactly."

    return fetch_url_tavily(url)

tools = [
    Tool(
        name="Academic_Search",
        func=lambda q: str(search_arxiv(q)),
        description="searches ArXiv. Returns a list of papers with 'title' and 'link'. Use the 'link' for reading."
    ),
    Tool(
        name="Web_Search_Tavily",
        func=search_web_tavily,
        description="searches the web using Tavily. Input must be a simple keyword string. Use this as an alternative to Web_Search."
    ),
    Tool(
        name="Web_Search",
        func=search_web,
        description="searches the web. Input must be a simple keyword string."
    ),
    Tool(
        name="Content_Reader_Tavily",
        func=safe_content_reader_tavily,
        description="Reads a webpage using Tavily. Input MUST be a valid HTTP URL (e.g., https://arxiv.org/...). DO NOT use article titles. Use this as an alternative to Content_Reader."
    ),
    Tool(
        name="Content_Reader",
        func=safe_content_reader, #Check input
        description="Reads a webpage. Input MUST be a valid HTTP URL (e.g., https://arxiv.org/...). DO NOT use article titles."
    )
]


class MistralChatParser(ReActSingleInputOutputParser):

    """
    Custom parser designed to handle the specific output patterns of Mistral-based models.
    It extracts 'Actions' for tool use and 'Final Answers' for user delivery.
    """
    def parse(self, text: str):
        if hasattr(text, 'content'):
            text = text.content
        text = text.strip()
        
        #Prevents the agent from hallucinating or predicting its own 'Observation'
        if "Action Input:" in text:
            # Locate the first occurrence of 'Observation' after the action input and cut it off
            parts = text.split("Action Input:")
            if len(parts) > 1:
                
                input_part = parts[1]
                if "Observation" in input_part:
                    clean_input_part = input_part.split("Observation")[0]
                    text = parts[0] + "Action Input:" + clean_input_part

        #If the model indicates a conclusion, stop the loop and return the result to the user
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        
        # Use regex to find the 'Action:' line which specifies the tool name
        action_match = re.search(r"Action:\s*(.*?)(?:\n|$)", text)
        if action_match:
            action_line = action_match.group(1).strip()
            chosen_tool = None
            tool_input = None
            
            # Match the action line against the list of available tools
            for tool in tools:
                if tool.name in action_line:
                    chosen_tool = tool.name
                    #Check if the model mistakenly included the input on the same line as the action
                    potential_input = action_line.replace(tool.name, "").strip()
                    if potential_input: # Clean up any extra parentheses or quotes if the input was merged with the tool name
                        tool_input = potential_input.strip('"').strip("'")
                    break
            
            #Extract the tool input (query) from the 'Action Input:' line if not captured previously.
            if chosen_tool and not tool_input:
                input_match = re.search(r"Action Input:\s*(.*?)(?:\n|$)", text)
                if input_match:
                    raw_input = input_match.group(1).strip()
                    tool_input = raw_input.split("(")[0].strip()

            # Small LLMs sometimes output tool inputs in JSON format (e.g., {"query": "text"}) instead of raw strings. This block extracts the pure value.
            if tool_input and "{" in tool_input:
                try: # Regex to find values after colons, specifically looking for text inside quotes
                    match = re.search(r':\s*"([^"]+)"', tool_input)
                    if match:
                        tool_input = match.group(1)
                    else:
                        tool_input = tool_input.replace("{", "").replace("}", "").replace('"query":', "").strip()
                except:
                    """
                    This is a conscious error-tolerance strategy designed to protect
                    the Agent Loop from crashing due to unpredictable formatting
                    ensuring a graceful fallback instead of a system failure
                    """
                    pass
            
            if tool_input is None: tool_input = "" #Ensure no nulls and remove any lingering surrounding quotes
            tool_input = tool_input.strip('"').strip("'")
            
            if chosen_tool:
                return AgentAction(tool=chosen_tool, tool_input=tool_input, log=text)

        return AgentFinish(return_values={"output": text}, log=text)

system_template = """You are a helpful research assistant.
You have access to: {tool_names}

FORMAT INSTRUCTIONS:
--------------------
You MUST use this format:

Thought: Do I need to use a tool? Yes
Action: [Tool Name]
Action Input: [Input]
(STOP HERE! Wait for Observation)

EXAMPLE WORKFLOW (Follow this pattern):
---------------------------------------
Question: Read the paper "Attention Is All You Need"
Thought: I need to find the link first.
Action: Academic_Search
Action Input: Attention Is All You Need
Observation: [{{'title': 'Attention Is All You Need', 'link': 'https://arxiv.org/abs/1706.03762'}}]
Thought: I found the paper. Now I will use the LINK to read it.
Action: Content_Reader
Action Input: https://arxiv.org/abs/1706.03762
Observation: The paper discusses...
Final Answer: The paper proposes the Transformer model...
---------------------------------------

CRITICAL RULES:
1. 'Action Input:' must be on a NEW LINE.
2. If using Content_Reader, Input MUST be a URL (starting with http), NOT a title.
3. NEVER use JSON.
"""

human_template = """Question: {input}
Thought: {agent_scratchpad}"""

"""
Define the conversational structure by combining the System instructions
and the Human's input into a unified prompt template
"""
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

#This section defines the execution flow using the pipe (|) operator.
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        "tool_names": lambda x: ", ".join([t.name for t in tools]),
    }
    | prompt
    | llm
    | MistralChatParser()
)

"""
The Executor is the orchestrator that manages the iterative loop:
It runs the agent, calls the tools, and repeats until a final answer is reached.
"""
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=5
)

if __name__ == "__main__":
    print("Agent Started! (Write 'exit' to quit)")
    while True:
        user_input = input("Question: ")
        if user_input.lower() == "exit": break
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"Answer: {response['output']}")
        except Exception as e:
            print(f"ERROR: {e}")