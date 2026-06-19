# Blog Writing Agent

An intelligent, multi-agent AI system built with LangGraph, LangChain, Groq, and Streamlit. This application researches, plans, and writes high-quality technical blogs autonomously, based on a given topic.

## Features
- **Multi-Agent Architecture**: Built with LangGraph, separating the generation into specialized nodes (Router, Research, Orchestrator, Worker, Reducer).
- **Automated Web Research**: Leverages Tavily to fetch up-to-date and relevant technical evidence for "hybrid" or "open book" blog modes.
- **Robust Rate Limiting**: Includes retry mechanisms for the Groq API to gracefully handle free-tier API limits during parallel generation.
- **Interactive UI**: A Streamlit frontend that streams the graph's progress in real-time.
- **Session Persistence**: Saves the blog's Markdown file along with a JSON file containing the generation metadata (Plan, Evidence, Logs). Past blogs can be quickly reloaded directly from the sidebar.

## Prerequisites
You will need API keys for the following services:
- **[Groq](https://console.groq.com/keys)** (for running the LLMs)
- **[Tavily](https://app.tavily.com/home)** (for web search and research)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd Blog-writer-agent
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv myenv
   # On Windows
   myenv\Scripts\activate
   # On macOS/Linux
   source myenv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your Environment Variables:**
   Create a `.env` file in the root of the project and add your API keys:
   ```env
   GROQ_API_KEY="your-groq-api-key"
   TAVILY_API_KEY="your-tavily-api-key"
   ```

## Usage

Start the Streamlit application:

```bash
streamlit run blog-agent-frontend.py
```

- Open the provided local URL (usually `http://localhost:8501`) in your browser.
- Enter a technical topic in the sidebar.
- Click **Generate Blog** and watch the agents research, outline, and write your post!
- You can find all previously generated blogs under the **Past blogs** section in the sidebar.

## Project Structure
- `blog-agent-backend.py`: The core LangGraph state graph defining the multi-agent logic, LLMs, and research tools.
- `blog-agent-frontend.py`: The Streamlit web interface that hooks into the graph and streams updates to the user.
- `requirements.txt`: Project dependencies.
