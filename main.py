from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot_logic import route_query
from generate_plan import generate_financial_plan
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

app = FastAPI()

# ✅ Add allowed origins for your local + hosted frontends/backends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # local React frontend
        "http://localhost:3002",  # local Node backend
        "https://finalyear-db-server.onrender.com",  # hosted Node backend
        "https://final-year-project-frontend-b1n7.onrender.com",  # hosted React frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- ROUTES -----------------

class ChatRequest(BaseModel):
    message: str
    userId: str


@app.post("/chatbot")
async def chatbot_handler(req: ChatRequest):
    user_query = req.message
    user_id = req.userId
    print("Chatbot query:", user_query)
    print("User ID:", user_id)
    response = route_query(user_query, user_id)
    return {"response": response}


class GenerateRequest(BaseModel):
    userId: str


@app.post("/generate")
async def generate_plan(request: GenerateRequest):
    try:
        print("Generating plan for User ID:", request.userId)
        user_id = request.userId
        result = generate_financial_plan(user_id)
        return {"generated_text": result}
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}

# ----------------- ENTRY POINT -----------------
if __name__ == "__main__":
    # ✅ Correct way to run locally
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
