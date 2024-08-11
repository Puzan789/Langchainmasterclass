
### I followed this YouTube guide: [LangChain Masterclass](https://youtu.be/yF9kGESAi3M?si=CyHHEHCvF6jbvGTG)
## Components

### 1. Retrieval-Augmented Generation (RAG)

**Retrieval-Augmented Generation (RAG)** enhances LLM outputs by integrating data retrieval mechanisms. It improves relevance, reduces hallucinations, and allows the use of external data without custom training. RAG implementations include:

- **Simple RAG**: Basic retrieval and response generation.
- **Simple RAG with Memory**: Adds conversation context.
- **Branched RAG**: Retrieves from multiple sources.
- **HyDe**: Uses hypothetical answers to guide retrieval.
- **Adaptive RAG**: Adjusts strategies based on query type.
- **Corrective RAG (CRAG)**: Refines responses with self-grading.
- **Self-RAG**: Ensures high accuracy with self-reflection.
- **Agentic RAG**: Combines RAG with agent-based problem-solving.

### 2. Agents

**Agents** are autonomous entities that decide and execute a sequence of actions dynamically based on the context. Unlike Chains, Agents do not follow a predefined sequence, making them more flexible and adaptable. Types of Agents include:

- **ZeroShotAgents**: Handle tasks without prior context.
- **ConversationalAgents**: Maintain context across interactions.
- **Task-Oriented Agents**: Focus on specific goals using a combination of tools and chains.

**Example Usage:**

```python
from langchain.agents import create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.google_finance import GoogleFinanceQueryRun

# Initialize the LLM and tool
llm = ChatOpenAI(model="gpt-3.5-turbo")
finance_tool = GoogleFinanceQueryRun()

# Create the agent
agent = create_openai_tools_agent(llm, tools=[finance_tool])

# Execute the agent
result = agent.invoke({"input": "What is the latest stock price of Google?"})
print(result)
```

### 3. Tools

**Tools** are interfaces that allow Agents, Chains, or LLMs to interact with external services or data sources. They can be predefined or custom-built, providing the functionality needed to perform specific tasks such as querying databases or executing calculations.

- **Predefined Tools**: Built-in tools available in LangChain (e.g., Google Finance, Wikipedia Query).
- **Custom Tools**: User-defined functions tailored for specific needs.

**Example Usage:**

```python
from langchain.tools import SimpleTool

@tool
def calculate_sum(a: int, b: int) -> int:
    return a + b

result = calculate_sum(10, 20)
print(result)  # Output: 30
```

### 4. Chains

**Chains** are sequences of operations that link together different tasks into a cohesive workflow. Chains can be simple or complex, and they allow developers to build modular and reusable pipelines for LLM-powered applications.

- **Sequential Chains**: Execute tasks in a predefined order, where the output of one task is the input for the next.
- **Parallel Chains**: Execute multiple tasks simultaneously, combining their outputs as needed.
- **Conditional Chains**: Use branching logic to determine which path to take based on specific conditions.

**Example Usage:**

```python
from langchain.chains import SimpleChain

def lower_case(text):
    return text.lower()

def add_greeting(text):
    return f"Hello, {text}!"

# Define a simple sequential chain
chain = SimpleChain([lower_case, add_greeting])

result = chain.run("WORLD")
print(result)  # Output: Hello, world!
```

### 5. Prompts

**Prompts** are templates that guide LLMs on how to respond to user inputs. They can be static or dynamic, with placeholders that are filled based on the context or user input.

- **Static Prompts**: Predefined text that the LLM uses as is.
- **Dynamic Prompts**: Include variables or placeholders that are filled with specific data at runtime, allowing for more flexible and context-aware responses.

**Example Usage:**

```python
from langchain.prompts import PromptTemplate

template = "Translate the following from English to French: '{text}'"
prompt = PromptTemplate(template)
result = prompt.format(text="Hello, world!")
print(result)  # Output: Translate the following from English to French: 'Hello, world!'
```
