
import argparse
import openai
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, required = True)
parser.add_argument("--key", type = str, required = True)
parser.add_argument("--count", type = int, required = True)
args = parser.parse_args()

gpt_35_model = "gpt-3.5-turbo-0125"
gpt_4_model = "gpt-4-1106-preview"
retry_limit = 5

openai.api_key = args.key
model = gpt_35_model if args.model == "3.5" else gpt_4_model

def invoke_openai(messages, model = gpt_35_model, temperature = None, max_token = None, response_format = None, stream = False):
    for _ in range(retry_limit):
        try:
            params = {
                "model": model,
                "messages": messages,
                "stream": stream
            }
            if max_token is not None: params["max_tokens"] = max_token
            if temperature is not None: params["temperature"] = temperature    
            if response_format is not None: params["response_format"] = response_format
            
            response = openai.ChatCompletion.create(**params)
            return response
        except Exception as exc:
            print(f"[ERROR] gpt failed: {exc}")
            print("[WARNING] retrying...")
    
    print("[ERROR] gpt completely failed")
    print(f"[ERROR] {messages}")
    
    return None

def call_openai_block(messages, model = gpt_35_model, temperature = None, max_token = None, response_format = None):
    completion = invoke_openai(messages, model, temperature, max_token, response_format, False)    
    if completion is None: return "error"
        
    return completion.get("choices")[0].get("message").get("content")

async def call_openai_stream(messages, model = gpt_35_model, temperature = None, max_token = None, response_format = None):
    completion = invoke_openai(messages, model, temperature, max_token, response_format, True)
    
    if completion:
        for chunk in completion:
            msg = chunk["choices"][0]["delta"]
            
            if msg:
                msg = msg["content"]
                yield msg
    else:
        for msg in ["error"]:
            yield msg

messages = [
    { "role": "system", "content": """
        This meeting is technincal interview for screening fullstack developer at the company `Factored.ai`.
        The provided meeting transcription has errors because it's transcribed by AI.
        You must find errors in transcription, reason out similarly pronouncing words or phrases that can replace the errors, and understand the meeting context correctly.
        The keywords (React, Javascript, C++, Vue.js) might be helpful for you to guess.
        Return JSON as {"you": <Your best next speaking>}
    """},
    { "role": "user", "content": """
        You are the candidate in job interview.
        The following text delimited by three backticks is meeting transcription until now.
        Return your best next speaking that is professional.
        Don't aplogize. Don't mention or explain about the transcription errors or your misunderstanding due to it..
        ```
        interviewer-> so let's dive in two the next phase of our interview it'll be a short technical questions okay?
        you-> yes sir please i am ready
        interviewer-> good the first, what is the we book in react to jaiss? can you tell me what we book functions you know and what experience do yo have?
        you-> yes, sure. err
        interviewer-> hmm?
        you-> way books in React.js are a powerful feature that allows developers to build reusable logic. I have experience implementing way books for real time data fetching and handling asynchronous operations efficiently. One specific example is using web hooks to trigger updates in the UI when new data is available without the need for manual refresh. This not only enhances user experience but also improves the overall performance of the application.
        interviewer-> cool so what is the name of the function you mentioned?
        you-> oh, well... er...
        interviewer-> okay, that's fine. and for the next how can you like i mean how can you dump sequel from streaming data like from google gcs?
        ```
    """}
]

total_dur = 0

for i in range(args.count):
    tick = time.time()
    answer = call_openai_block(model = model, messages = messages, response_format = {"type": "json_object"}, temperature = 0.001, max_token = 200)
    dur = time.time() - tick
    
    total_dur += dur
    print("[ITER-{:03}]: {:.4f}s".format(i + 1, dur))

print(f"answer: f{answer}")
print("avg: {:.4f}s".format(total_dur / args.count))
