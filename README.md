# Enhanced Multi-Agent System (Community Edition --Pl use WSL if running on Windows:-   wsl --install -d Ubuntu)

A revolutionary approach to AI agent orchestration with minimal code and maximum capability. This system provides powerful multi-agent workflows through simple JSON configuration rather than complex Python code.

Sort of extending Anthropics MCP for Agents.  While MCP primarily addresses application-to-model communication, our system enables sophisticated agent-to-agent interactions with memory sharing and workflow orchestration across complex networks of specialized agents.

## Why This System Stands Out

While frameworks like LangChain, LangGraph, and Semantic Kernel require hundreds of lines of Python code to orchestrate complex agent interactions, this system achieves the same capabilities through **declarative JSON configuration**:

- What would take 500+ lines of Python code in LangChain can be done in ~300 lines of JSON here
- No callbacks, chain composition, or complex error handling required
- Simple yet powerful orchestration of sophisticated agent workflows
- Enterprise-ready with batteries included

## Core Features

### 1. Memory Management
- **Persistent SQL-based memory** shared across agents and sessions
- **No unnecessary vector embeddings** for structured data workflows
- Memory context automatically added to prompts with zero configuration

### 2. Dynamic Agent Behavior
- Agents can **dynamically choose their next actions** based on task requirements
- **Branching workflows** that adapt based on agent decisions
- **Decision-based delegation** to specialized agents

### 3. Structured Output Parsing
- **JSON schema validation** for structured agent outputs
- **Markdown section validation** for document outputs
- **Automatic extraction** and structuring of agent responses






# Sophisticated Workflows with Simple Configuration

The power of this system is best demonstrated by the included `extremely_advanced_cyber_agentic_workflow.json`, which implements a comprehensive security analysis pipeline with:

- Multi-stage analysis of session data
- Behavior pattern recognition
- Threat correlation across multiple signals
- Dynamic response selection based on findings
- Business impact analysis
- Executive reporting
- Detailed remediation planning

**All of this is achieved with a single JSON configuration file** - no Python coding required beyond the base system. Implementing an equivalent workflow in LangChain would require hundreds of lines of Python code with complex chain compositions, custom callbacks, and extensive error handling.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/RahulModak74/enhanced_agents.git
cd enhanced_agents

# Install basic dependencies
pip install jsonschema

# Make the main script executable
chmod +x universal_main.py

# Set up your API key in config.py
# Get a free API token for DeepSeek on openrouter and add it to config.py


python3 universal_main.py --workflow cust_jour_workflow.json customer_journey.csv

python3 universal_main.py --workflow extremely_advanced_cyber_agentic_workflow.json journey.csv


```

## Usage Options

### Interactive Mode

```bash
./universal_main.py
```

This launches an interactive CLI where you can:

- Create agents: `create agent1 [model] [type]`
- Run agents: `run agent1 What is the risk score? [file] data.csv [memory:risk] [format:json]`
- Execute workflows: `workflow example_workflow.json`
- Access memory: `memory get risk_assessment` or `memory list`
- List agents: `list`
- Get agent outputs: `get agent1`

### Workflow Mode

```bash
./universal_main.py --workflow example_workflow.json
```

This executes a pre-defined workflow without entering the interactive shell.

## Advanced Capabilities (COMPONENT Directory)

While the core system is intentionally minimal, the COMPONENT directory contains powerful extensions:

### SQL Tool Integration
- Query relational databases directly from agents
- Perfect for enterprise data stored in traditional formats
- Import/export capabilities for tabular data

### HTTP/API Tool
- Make HTTP requests to external services
- Support for all methods, authentication, and file uploads
- Built-in rate limiting and error handling

### Vector Database Integration
- Semantic search capabilities for unstructured data
- FAISS-based embedding storage and retrieval
- Use only when you genuinely need semantic matching

### Planning Tools
- Chain of Thought (CoT) for structured reasoning
- ReAct pattern (Reasoning + Acting) for complex problem-solving
- Task planning with goals and subtasks

These extensions are provided separately to maintain the core system's simplicity. Use them only when your specific workflow requires these capabilities, rather than adding unnecessary complexity by default.

## The Philosophy: Right Tool for Each Job

A key advantage of this system is its pragmatic approach to tool selection:

- Most enterprise data remains in structured, relational formats (from SAP, CRM, etc.)
- Vector databases are valuable only for specific use cases (semantic search, multimodal data)
- SQL queries are faster and more precise for structured data

The system lets you use the right tool for each specific task rather than forcing everything into embeddings or complex chains.

## Project Structure

```
enhanced_agents/
├── __init__.py             # Package initialization
├── config.py               # Configuration settings
├── memory_manager.py       # Memory persistence system
├── output_parser.py        # Structured output parsing
├── agent.py                # Base agent implementation
├── dynamic_agent.py        # Dynamic decision-making agent
├── agent_system.py         # Agent orchestration system
├── cli.py                  # Command-line interface
├── main.py                 # Main application entry point
├── example_workflow.json   # Example workflow configuration
├── extremely_advanced_cyber_agentic_workflow.json  # Advanced security workflow
└── COMPONENT/              # Advanced tool integrations
    ├── sql_tool.py         # SQL database integration
    ├── http_tool.py        # HTTP/API tool
    ├── vector_db.py        # Vector database integration 
    ├── planning_tool.py    # Planning and reasoning tools
    └── README.md           # Component-specific documentation
```

## Workflow Configuration

Workflows are defined as JSON arrays of agent execution steps:

```json
[
  {
    "agent": "agent1",
    "content": "Analyze this data...",
    "file": "data.csv",
    "output_format": {
      "type": "json",
      "schema": {
        "score": "number",
        "reasoning": "string"
      }
    },
    "memory_id": "analysis"
  },
  {
    "agent": "dynamic_agent",
    "type": "dynamic",
    "initial_prompt": "Decide what to do next...",
    "readFrom": ["agent1"],
    "memory_id": "analysis",
    "actions": {
      "action1": {
        "agent": "agent2",
        "content": "Follow-up task..."
      },
      "action2": {
        "agent": "agent3",
        "content": "Alternative task..."
      }
    }
  }
]
```

## Feature Comparison with LangChain and Community Version

This system achieves approximately feature parity with LangChain's agent orchestration capabilities, while requiring significantly less code:

**What it has:**

1. **Core Agent Orchestration (90%)** - Handles sequential agent execution, context sharing, and dynamic routing
2. **Memory Management (85%)** - SQLite-based persistent storage across sessions with comprehensive retrieval
3. **Structured Output Parsing (80%)** - JSON schema validation and structured output extraction
4. **Dynamic Behavior (75%)** - Dynamic agents with path selection and information routing
5. **Command Line Interface (90%)** - Full-featured interactive CLI
6. Premium version has research lab grade  features for memory cognition and optimization.

**Advanced Capabilities (Available in COMPONENT Directory):**

1. **Tool Integration** - SQL, HTTP, and Vector DB tools for when you need them
2. **Planning & Reasoning** - CoT, ReAct, and structured planning capabilities
3. **Vector Operations** - FAISS-based embedding storage and retrieval

For routine agent orchestration where you primarily need agents to work together, share context, make decisions, and maintain state, this system handles most use cases with dramatically simpler configuration than alternatives.

## Advanced Component Example: Customer Journey Analysis

Our COMPONENT directory includes advanced tools that perfectly complement the core system when needed. The following example demonstrates how these tools work together in a complete customer journey analysis workflow:

```json
[
  {
    "agent": "journey_planner",
    "content": "Analyze the customer_journey.csv file and create a detailed plan...",
    "file": "customer_journey.csv",
    "tools": ["planning:create_plan", "planning:chain_of_thought"],
    "output_format": {
      "type": "json",
      "schema": {
        "data_overview": "string",
        "analysis_plan": {
          "goal": "string",
          "subtasks": ["string"]
        },
        "key_metrics_to_track": ["string"],
        "reasoning": "string"
      }
    },
    "memory_id": "journey_analysis"
  },
  // Additional agents in the workflow...
]
```

This workflow demonstrates how:

Planning tools provide structured reasoning and task planning
SQL tools analyze structured customer data directly
Memory management shares insights between specialized agents
Dynamic routing selects optimization strategies based on analysis

While other frameworks would require:

Complex vector embedding setup for memory
Custom Python code for each analysis step
Specialized callbacks and error handling
Chain composition code for agent interaction

Our system accomplishes all of this with declarative JSON configuration.
The complete workflow includes specialized agents for journey analysis, segment analysis, friction identification, optimization strategy, dynamic decision-making, ROI calculation, and executive reporting - all connected through a simple, readable JSON configuration file.

You can run this workflow with:
python3 main.py --tool-workflow customer_journey_planning_workflow.json
This demonstrates the power of our approach - handling complex multi-agent workflows without sacrificing simplicity.

PROFESSIONAL EDITION:

Has multiple advanced features like below:
Key Cognitive Architecture Components:


Memory Systems Framework The Memory Systems Framework provides advanced memory patterns that enable agents to store, retrieve, and reason over complex information:

Key Capabilities

Federated Knowledge Graphs: Distributes knowledge across specialized nodes for domain-specific processing and cross-domain integration.

Hierarchical Memory Architecture: Implements a multi-tier memory system with working memory, context-sensitive retrieval, and intelligent promotion/demotion of information based on relevance.

Memory Synchronization Protocol: Enables coordination between multiple agents through publisher-subscriber patterns with consistency verification.

Research Integration: Automatically extracts entities, relations, and concepts from research data and incorporates them into the appropriate memory structures.

Recommendation Engine: Analyzes memory patterns to identify knowledge gaps, promising research areas, and emergent knowledge clusters.

Cognitive Planning Framework The Cognitive Planning Framework implements advanced reasoning patterns that mirror how humans approach complex problems:

Key Capabilities

Tree of Thoughts: Explores multiple solution paths through branching exploration, iterative refinement, and comparative evaluation.

Metacognitive Reflection: Enables agents to analyze their own reasoning processes, detect biases, explore counterfactuals, and self-optimize.

Multi-agent Debate Protocol: Facilitates structured debate between competing perspectives with moderation, rebuttals, and synthesis.

Adaptive Cognition: Dynamically selects and applies the most appropriate cognitive architecture based on problem characteristics.

## License

SSPL
