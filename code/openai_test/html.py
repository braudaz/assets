
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
        For the nested fields, don't reflect their nested structure in output JSON. Just analyze them and include all relevant fields in JSON as instructed below.
        The structure of JSON is:

        {
            "text-fields": [
                <Label description for this text field.>,
                ...
            ],
            "option-fields": [
                {
                    "label": <Label description for this option field.>,
                    "choices": [
                        <choice>,
                        ...
                    ]
                },
                ...
            ],
            "check-fields": [
                {
                    "label": <Label description for this check field.>,
                    "choices": [
                        <choice>,
                        ...
                    ]
                },
                ...
            ]
        }
    """},
    { "role": "user", "content": """
        
        Analyze the following HTML snippet (delimited by three backticks) for a submission form and return JSON providing insight of all fields to fill.
        "text-fields" are the fields requiring text input.
        "option-fields" are the fields requiring only one choice from many.
        "check-fields" are the fileds allowing multiple selection.
        

        ```
        <div><div><div><div><div><h3>Autofill from resume</h3><p>Upload your resume here to autofill key application fields.</p></div><div><button>Upload file</button></div></div></div><div><span>Drop your resume here!</span></div><div><span>Parsing your resume. Autofilling key fields...</span></div></div></div><div><div><div><div><span>This job has application limits</span></div></div><div><div><p>Please Note: we have set up limits for applications across roles.  Candidates may not apply more than 5 times in any 180 day span.</p></div></div></div></div><div><div><div><label>Name</label></div><div><label>Email</label></div><div><label>Resume</label><div><div><button><span><span>Upload File</span></span></button><p>or drag and drop here</p></div></div></div><div><label>Phone</label></div><div><label>Cover Letter</label><div><div><button><span><span>Upload File</span></span></button><p>or drag and drop here</p></div></div></div><div><label>Do you have experience working higher education and/or K-12 accounts?</label><div><button>Yes</button><button>No</button></div></div><div><label>Are you able to work from our San Francisco HQ 3 days per week? 

        *Please do not apply if not open to relocation or unable to work from SF </label><div><button>Yes</button><button>No</button></div></div><fieldset><label>Are you authorized to work lawfully in the US?</label><div><span></span><label>Other</label></div><div><span></span><label>No</label></div><div><span></span><label>Yes</label></div></fieldset><div><label>LinkedIn profile</label></div><div><label>Additional Information</label></div><div><label>When can you start a new role?</label></div><fieldset><label>How did you hear about OpenAI?</label><div><span></span><label>Other source (indicate in Additional Information)</label></div><div><span></span><label>Wonder Women Tech</label></div><div><span></span><label>Underdog</label></div><div><span></span><label>Twitter</label></div><div><span></span><label>The Plug - TP Insights</label></div><div><span></span><label>Techqueria</label></div><div><span></span><label>OpenAI Employee</label></div><div><span></span><label>OpenAI Blog</label></div><div><span></span><label>LinkedIn</label></div><div><span></span><label>Jopwell</label></div><div><span></span><label>Hacker News</label></div><div><span></span><label>Glassdoor</label></div><div><span></span><label>GirlGeekX</label></div><div><span></span><label>DiversityX</label></div><div><span></span><label>Conference - other</label></div><div><span></span><label>Conference - Scale Transform 2021</label></div><div><span></span><label>Conference - NeurIPS</label></div><div><span></span><label>Conference - ICPR</label></div><div><span></span><label>Conference - ICML</label></div><div><span></span><label>Conference - ICLR</label></div><div><span></span><label>Conference - AI for Good</label></div><div><span></span><label>Breakout List</label></div><div><span></span><label>80,000 Hours</label></div></fieldset></div></div><div><div><div><div><h2>Diversity Survey</h2><div><p><em>This optional survey helps us evaluate our diversity and inclusion efforts.  Participation is voluntary and refusal to submit the survey will not affect your job application.  The answers to these questions are not seen on an individualized basis and your submission will only be used to assess our diversity and inclusion efforts.</em></p></div></div><fieldset><label>What is your current age?</label><div><span></span><label>Under 30</label></div><div><span></span><label>30-39</label></div><div><span></span><label>40-49</label></div><div><span></span><label>50-59</label></div><div><span></span><label>60 or older</label></div><div><span></span><label>I prefer not to answer</label></div></fieldset><fieldset><label>What is your gender identity?</label><div><span></span><label>Man</label></div><div><span></span><label>Woman</label></div><div><span></span><label>Non-Binary</label></div><div><span></span><label>Another Gender Identity</label></div><div><span></span><label>I prefer not to answer</label></div></fieldset><fieldset><label>Do you identify as transgender?</label><div><span></span><label>Yes</label></div><div><span></span><label>No</label></div><div><span></span><label>I prefer not to answer</label></div></fieldset><fieldset><label>How do you identify your sexual orientation? Please select all that apply.</label><div><span></span><label>Bisexual</label></div><div><span></span><label>Lesbian</label></div><div><span></span><label>Gay</label></div><div><span></span><label>Queer</label></div><div><span></span><label>Heterosexual / straight</label></div><div><span></span><label>Other</label></div><div><span></span><label>I prefer not to answer</label></div></fieldset><fieldset><label>Which ethnicity(ies) do you identify with? Please select all that apply.</label><div><span></span><label>Asian or Asian American</label></div><div><span></span><label>Black or African American</label></div><div><span></span><label>Hispanic or Latine</label></div><div><span></span><label>Indigenous or Native American</label></div><div><span></span><label>Native Hawaiian or Other Pacific Islander</label></div><div><span></span><label>White</label></div><div><span></span><label>Other</label></div><div><span></span><label>I prefer not to answer</label></div></fieldset><fieldset><label>Which of the following communities do you belong to? Please select all that apply.</label><div><span></span><label>Person with disability</label></div><div><span></span><label>Neurodiverse</label></div><div><span></span><label>Veteran</label></div><div><span></span><label>Parent</label></div><div><span></span><label>Refugee or immigrant</label></div><div><span></span><label>None of the above</label></div><div><span></span><label>I prefer not to answer</label></div></fieldset></div></div></div><button><span>Submit Application</span></button>
        ```
    """}
]

total_dur = 0

for i in range(args.count):
    tick = time.time()
    answer = call_openai_block(model = model, messages = messages, temperature = 0.001)
    dur = time.time() - tick
    
    total_dur += dur
    print("[ITER-{:03}]: {:.4f}s".format(i + 1, dur))

print(f"answer: f{answer}")
print("avg: {:.4f}s".format(total_dur / args.count))
