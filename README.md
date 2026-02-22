# 🎓 Academic Search Agent

Academic Search Agent is an autonomous AI research assistant built on LangChain and the ReAct (Reasoning and Acting) framework.
It is designed to automate academic literature reviews by intelligently searching through ArXiv and web sources, synthesizing complex information into structured academic insights.

## ✨ Key Features

* Autonomous Research: The agent independently decides which tools (ArXiv, Web Search) to use based on the complexity of the query.

* Academic-First Approach: Deep integration with the ArXiv API for real-time access to the latest scientific papers.

* Hybrid Architecture: Seamlessly combines a FastAPI backend with a Streamlit frontend for a responsive user experience.

* Open-Source LLM Power: Leverages Hugging Face Inference API (Mistral-7B/Zephyr) to provide high-quality reasoning without high costs.

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

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
