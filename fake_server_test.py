import asyncio
import time
from server import app, ChatCompletionRequest, Message, chat_completions

async def main():
    print("--- 1. Initializing lifespan (simulating server boot) ---")
    # Simulate FastAPI lifespan startup
    async with app.router.lifespan_context(app) as context:
        print("✅ Lifespan context initialized successfully!")
        
        # Construct fake completions request
        req = ChatCompletionRequest(
            messages=[Message(role="user", content="Reply with the number 42.")],
            max_tokens=512,
            do_sample=False,
            stream=False
        )
        
        print("\n--- 2. Sending fake chat completions request ---")
        t0 = time.time()
        try:
            resp = await chat_completions(req)
            print("✅ Success!")
            print(f"Response: {resp.body.decode('utf-8')}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            
if __name__ == "__main__":
    asyncio.run(main())
