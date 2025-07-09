# Model Context Protocol Project
This MCP client-server project includes a server (worker_server.py, below) implemented with tools that function the same as the three agents in 'supervisor_worker_4.py' (Supervisor_worker_agent_project repo/supervisor_worker_script directory). The project uses the mcp package (from pip), often referred to as 'FastMCP', from the Anthropic/Claude platform.

### Host/MCP client script: mcp_chatbot.py
- load and run servers
- select and use appropriate MCP tool based on user query
- generate response to user based on tool output
- manage query input and response process
- manage script sequencing (async)

### Research paper search server: research_server.py
- find and retrieve papers from the arXiv.org e-Print archive
- 'search_papers': search for papers on a topic
- 'extract_info': return information and summary on a specific paper ID

### Worker server: worker_server.py
- tools that replicate the function of worker agents from a LangChain/LangGraph server/worker agent framework.
- 'predict_tool': predict iris species from petal and septal measurements
  - ML prediction model: iris_model.pkl (RandomForest model trained on iris dataset)
- 'wikipedia_search': search Wikipedia
- 'tavily_search': search the web

### Config file to load servers: server_config.json
- information needed to load and run servers:
  - the two local servers above
  - two remote servers: 'filesystem' and 'fetch'
    - (https://github.com/modelcontextprotocol/servers)


Project based on DeepLearning short course: "MCP: Build Rich-Context AI Apps with Anthropic"  
(https://learn.deeplearning.ai/courses/mcp-build-rich-context-ai-apps-with-anthropic/lesson/ccsd0/why-mcp)
- implemented in normal Python (not notebook)
- created new server: worker_server.py
- loads and runs new server
- changed LLM in 'predict_tool' to Claude
