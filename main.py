from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot_logic import route_query 
from generate_plan import generate_financial_plan
app = FastAPI()
import os
from dotenv import load_dotenv
load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://finalyear-db-server.onrender.com",
        "https://final-year-project-frontend-b1n7.onrender.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    userId: str

@app.post("/chatbot")
async def chatbot_handler(req: ChatRequest):
    user_query = req.message
    print(user_query)
    user_id = req.userId
    print(user_id)
    response = route_query(user_query,user_id)
    return {"response": response}

class GenerateRequest(BaseModel):
    userId: str

@app.post("/generate")
async def generate_plan(request: GenerateRequest):
    try:
        print("User ID:", request.userId)
        user_id = request.userId
        result = generate_financial_plan(user_id)
        return {"generated_text": result}
        print(result)
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)




