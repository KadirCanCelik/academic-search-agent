# 🎓 Academic Search Agent

Academic Search Agent is an autonomous AI research assistant built on LangChain and the ReAct (Reasoning and Acting) framework.
It is designed to automate academic literature reviews by intelligently searching through ArXiv and web sources, synthesizing complex information into structured academic insights.

## ✨ Key Features

* **Autonomous Research:** The agent independently decides which tools (ArXiv, Web Search) to use based on the complexity of the query.

* **Academic-First Approach:** Deep integration with the ArXiv API for real-time access to the latest scientific papers.

* **Hybrid Architecture:** Seamlessly combines a FastAPI backend with a Streamlit frontend for a responsive user experience.

* **Open-Source LLM Power:** Leverages Hugging Face Inference API (Mistral-7B) to provide high-quality reasoning without high costs.

## 🧩 Project Structure
.  
├── app/  
│   ├── backend/        # Agent logic and data schemas  
│   ├── frontend/       # Streamlit UI implementation  
│   └── main.py         # FastAPI server entry point  
├── Dockerfile          # Container configuration  
├── requirements.txt    # Python dependencies  
└── README.md           # Project documentation  
└── start.sh            # Orchestration script to run Backend & Frontend concurrently  

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACEHUB_API_TOKEN` | Yes | Hugging Face Inference API token for the LLM |
| `TAVILY_API_KEY` | No | Tavily API key for web search. If set, Tavily is used instead of DuckDuckGo. Get a free key at https://app.tavily.com |

To use Tavily as the web search provider, set the `TAVILY_API_KEY` environment variable:

```bash
export TAVILY_API_KEY="tvly-your-api-key"
```

If `TAVILY_API_KEY` is not set, DuckDuckGo is used as the default web search fallback.

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🌐 Live Demo & Deployment Notes  

The project is deployed on Hugging Face Spaces. To overcome cloud environment limitations and IP restrictions, an ArXiv-Only optimization has been implemented for the live demo to guarantee consistent performance.
* **Link:** https://huggingface.co/spaces/kadircancelik/Academic-Search
