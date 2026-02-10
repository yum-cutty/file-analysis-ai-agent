import json
import os
import time
import requests
from typing import TypeAlias, cast
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
from groq.types.chat import ChatCompletion, ChatCompletionToolParam, ChatCompletionMessageToolCall, ChatCompletionAssistantMessageParam

# 3. api calling tools (functions)

# ################################################################ #
#                           GET API KEY                            #
# ################################################################ #

load_dotenv()
ai: Groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ################################################################ #
#                         STRUCTURE FORMAT                         #
# ################################################################ #

class Weather(BaseModel):
    temperature: float
    response: str

# ################################################################ #
#                        FUNCTION AI CALLS                         #
# ################################################################ #

weatherType: TypeAlias = dict[str, float | str | int]

"""
sample output:
{
    'time': '2026-02-07T08:15', 
    'interval': 900, 
    'temperature_2m': 11.6, 
    'wind_speed_10m': 14.1
}
"""

def get_weather(latitude: float, longitude: float) -> weatherType:
    """
    Fetches the current weather for the given latitude and longitude. Meant to be used as a tool by the AI model.
    
    :param latitude: latitude of the location
    :type latitude: float
    :param longitude: longitude of the location
    :type longitude: float
    :return: current weather data based on the given coordinates
    :rtype: weatherType
    """
    response: requests.Response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    # we return only the current weather for simplicity
    return response.json()["current"]

# ################################################################ #
#                          CALLBACK BY AI                          #
# ################################################################ #

toolType: TypeAlias = list[ChatCompletionToolParam]

tools: toolType = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Fetches the current weather for the given latitude and longitude.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},  # float is not a valid JSON schema type
                "longitude": {"type": "number"}, # float is not a valid JSON schema type
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False,
        },
        "strict": True,
    }
}]

# ################################################################ #
#                             API CALL                             #
# ################################################################ #

print("Sending request to Groq API...")
start_time: float = time.time()

# NOTE ai has no ability to call any of the function here
# NOTE the only thing it can do is figure out the function (tool) to call and provide the arguments
# NOTE the actual function calling is done outside of the ai call

# =========================
# first api call
# ========================

# this API call is to figure out the  arguments to add

first_request: ChatCompletion = ai.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        { "role": "system", "content": "You're a helpful weather assistant." },
        { "role": "user", "content": "What's the weather like in New York?" }
    ],
    tools=tools,
    max_tokens=4096,
    temperature=0.7,
    reasoning_effort="low",
)

# since we do not know the coordinates of New York, we let AI does the job of what we gonna put in
# extract the tool call and arguments

tool_calls: list[ChatCompletionMessageToolCall] | None = first_request.choices[0].message.tool_calls
assert tool_calls is not None, "Model did not return any tool calls"

tool_call: ChatCompletionMessageToolCall = tool_calls[0]
args = json.loads(tool_call.function.arguments)

# now we have the arguments, we can call the function ourselves

weather_data: weatherType = get_weather(args["latitude"], args["longitude"])

# =========================
# second api call
# ========================

# this API call is to get the final response with the response type we want

second_request: ChatCompletion = ai.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        { "role": "system", "content": 'You\'re a helpful weather assistant. Respond in JSON with this schema: {"temperature": float, "response": string}' },
        { "role": "user", "content": "What's the weather like in New York?" },
        cast(ChatCompletionAssistantMessageParam, first_request.choices[0].message.to_dict()),
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(weather_data),
        },
    ],
    response_format={"type": "json_object"},
    max_tokens=4096,
    temperature=0.7,
    reasoning_effort="low",
)

end_time: float = time.time()
print(f"Request completed in {end_time - start_time:.2f} seconds.\n")

# ################################################################ #
#                          PRINT RESULTS                           #
# ################################################################ #

response: str | None = second_request.choices[0].message.content
assert response is not None, "API returned no content"

weather = Weather(**json.loads(response))
print(weather.temperature)
print(weather.response)