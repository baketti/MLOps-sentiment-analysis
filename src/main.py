if __name__ == "__main__":
    from api.main import app
    import uvicorn
    import os

    uvicorn.run(
        app=app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 3000))
    )
