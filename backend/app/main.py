from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SKINORA API",
    description="AI-powered Skin Health Assessment System",
    version="1.0.0"
)

# Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to SKINORA API"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }