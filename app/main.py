from fastapi import FastAPI, HTTPException
from app.backend.models import QueryInput, AgentResponse
from app.backend.agent import agent_executor
import uvicorn

# Initialize FastAPI app with metadata for auto-generated documentation (Swagger)
app = FastAPI(
    title="Zephyr Research Agent API",
    description="Akademik ve Genel Araştırma Asistanı API Servisi",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """
    Health check endpoint to verify if the server is running
    """
    return {"status": "online", "message": "Research Agent API is running."}

@app.post("/ask", response_model=AgentResponse)
async def ask_agent(query_data: QueryInput):
    """
    Main endpoint that receives a query invokes the agent and returns the structured response
    """
    try:
        # Invoke the LangChain agent with the user's input
        response = agent_executor.invoke({"input": query_data.input})
        
        # Structure the internal agent logs into the defined Pydantic model
        return AgentResponse(
            output=response["output"],
            intermediate_steps=[str(step) for step in response.get("intermediate_steps", [])]
        )
    except Exception as e:
        # Provide detailed error message if the agent execution fails
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)