from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok", "service": "ml-service"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Остальные endpoints добавим позже
