import os
import json
import uuid
import time
import re
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextIteratorStreamer
from threading import Thread
import torch

app = FastAPI()

MODEL_ID = "OpenVINO/gemma-4-E4B-it-int8-ov"
print(f"Loading model {MODEL_ID}...")

ov_config = {
    "CACHE_DIR": "ov_cache",
    "INFERENCE_NUM_THREADS": "2",
    "NUM_STREAMS": "1",
}

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = OVModelForVisualCausalLM.from_pretrained(
    MODEL_ID, 
    device="CPU", 
    ov_config=ov_config,
    compile=True
)
print("Model loaded successfully.")

def parse_tool_calls(text):
    # Tool call format: <|tool_call>call:function_name{arg1:val1,arg2:val2}<tool_call|>
    # The template says: <|tool_call|>call:name{args}<tool_call|>
    # regex for <|tool_call|>call:name{args}<tool_call|>
    pattern = r"<\|tool_call\|?>call:(\w+)\{(.*?)\}<tool_call\|?>"
    matches = re.finditer(pattern, text)
    tool_calls = []
    for match in matches:
        name = match.group(1)
        args_str = match.group(2)
        
        args = {}
        if args_str.strip():
            # Naive parser for k1:v1,k2:v2 where v can be quoted
            # Example: command:"touch /root/test",timeout:10
            # Let's try to match key:value pairs
            # The values can be strings in <|"|>...<|"|> or just text
            items = re.findall(r'(\w+):(?:<\|"\|>(.*?)<\|"\|>|"(.*?)"|([^,]*))', args_str)
            for k, v_tag, v_quote, v_raw in items:
                v = v_tag if v_tag else (v_quote if v_quote else v_raw)
                v = v.strip()
                # Try to parse as JSON if it looks like it, else keep as string
                try:
                    args[k] = json.loads(v)
                except:
                    args[k] = v
        
        tool_calls.append({
            "id": "call_" + str(uuid.uuid4())[:8],
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args)
            }
        })
    return tool_calls

@app.post("/v3/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    tools = data.get("tools", None)
    stream = data.get("stream", False)
    model_name = data.get("model", "gemma-4-it")
    max_tokens = data.get("max_tokens", 1024)
    temperature = data.get("temperature", 0.0)
    extra_body = data.get("extra_body", {})
    chat_template_kwargs = extra_body.get("chat_template_kwargs", {})

    prompt = processor.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        **chat_template_kwargs
    )

    inputs = processor(prompt, return_tensors="pt").to(model.device)
    
    if stream:
        def generate():
            streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=False)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            created = int(time.time())
            completion_id = "chatcmpl-" + str(uuid.uuid4())
            
            full_content = ""
            for new_text in streamer:
                full_content += new_text
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            tool_calls = parse_tool_calls(full_content)
            if tool_calls:
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": tool_calls},
                        "finish_reason": "tool_calls"
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            else:
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
        outputs = model.generate(**generation_kwargs)
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        decoded = processor.decode(generated_ids, skip_special_tokens=False)
        
        tool_calls = parse_tool_calls(decoded)
        content = re.sub(r"<\|tool_call\|?>.*?<tool_call\|?>", "", decoded, flags=re.DOTALL).strip()
        
        response = {
            "id": "chatcmpl-" + str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content if content else None,
                    "tool_calls": tool_calls if tool_calls else None
                },
                "finish_reason": "tool_calls" if tool_calls else "stop"
            }],
            "usage": {
                "prompt_tokens": inputs.input_ids.shape[-1],
                "completion_tokens": len(generated_ids),
                "total_tokens": inputs.input_ids.shape[-1] + len(generated_ids)
            }
        }
        return JSONResponse(content=response)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
