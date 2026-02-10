import json
import os
import time
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
from groq.types.chat import ChatCompletion

# 2. api response structured handling

# ################################################################ #
#                           GET API KEY                            #
# ################################################################ #

load_dotenv()
ai: Groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ################################################################ #
#                         STRUCTURE FORMAT                         #
# ################################################################ #

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

# ################################################################ #
#                             API CALL                             #
# ################################################################ #

print("Sending request to Groq API...")
start_time: float = time.time()

request: ChatCompletion = ai.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        { "role": "system", "content": "Extract the event information. Respond in JSON with this schema: {\"name\": string, \"date\": string, \"participants\": [string}" },
        { "role": "user", "content": "Alice and Bob are going to a science fair on Friday." }
    ],
    response_format={
        "type": "json_object",
    },
    max_tokens=4096,
    temperature=0.7,
    reasoning_effort="low",
)

end_time: float = time.time()
print(f"Request completed in {end_time - start_time:.2f} seconds.\n")

# ################################################################ #
#                          PRINT RESULTS                           #
# ################################################################ #

response: str | None = request.choices[0].message.content
assert response is not None, "API returned no content"

calendar_event = CalendarEvent(**json.loads(response))
print(calendar_event.name)
print(calendar_event.date)
print(calendar_event.participants)