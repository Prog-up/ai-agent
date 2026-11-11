import threading
import time
import requests
import uvicorn
from server import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, loop="asyncio", http="h11", ws="none")

print("--- Starting Uvicorn in background thread ---")
t = threading.Thread(target=run_server, daemon=True)
t.start()

# Wait for server to start and warmup
print("Waiting 70 seconds for server startup and warmup...")
time.sleep(70)

# Send request
print("\n--- Sending HTTP request to Uvicorn ---")
try:
    resp = requests.post(
        "http://127.0.0.1:8001/v1/chat/completions",
        json={
            "model": "gemma",
            "messages": [{"role": "user", "content": "Reply with the number 42."}],
            "stream": False
        },
        timeout=30
    )
    print("✅ Status code:", resp.status_code)
    print("✅ Response:", resp.text)
except Exception as e:
    import traceback
    print("❌ Request failed:")
    traceback.print_exc()
