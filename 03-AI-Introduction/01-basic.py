import os
import time
from dotenv import load_dotenv
from groq import Groq
from groq.types.chat import ChatCompletion

# 1. basic api calling

# ################################################################ #
#                           GET API KEY                            #
# ################################################################ #

load_dotenv()
ai: Groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ################################################################ #
#                             API CALL                             #
# ################################################################ #

print("Sending request to Groq API...")
start_time: float = time.time()

request: ChatCompletion = ai.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        { "role": "system", "content": "You are an expert in English poetry." },
        { "role": "user", "content": "Write a short poem about the sea." }
    ],
    max_tokens=4096,
    temperature=0.7,
    reasoning_effort="low"
)

end_time: float = time.time()
print(f"Request completed in {end_time - start_time:.2f} seconds.\n")

# ################################################################ #
#                          PRINT RESULTS                           #
# ################################################################ #

print(request.choices[0].message.content)