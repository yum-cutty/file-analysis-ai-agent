import json
import os
import time
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

class Knowledge(BaseModel):
    answer: str
    source: str | int

# ################################################################ #
#                        FUNCTION AI CALLS                         #
# ################################################################ #

def get_knowledge(question: str) -> dict[str, list[dict[str, int | str]]]:
    with open("knowledge.json", "r") as f:
        return json.load(f)

# ################################################################ #
#                          CALLBACK BY AI                          #
# ################################################################ #

toolType: TypeAlias = list[ChatCompletionToolParam]

tools: toolType = [{
    "type": "function",
    "function": {
        "name": "get_knowledge",
        "description": "Get the answer to the user's question from the knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": { "type": "string" }
            },
            "required": ["question"],
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
        { "role": "system", "content": "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store." },
        { "role": "user", "content": "Is there any return/refund policy? if yes, what is the policy?" }
    ],
    tools=tools,
    max_tokens=4096,
    temperature=0.7,
    reasoning_effort="low",
)

# extract the tool call and arguments

tool_calls: list[ChatCompletionMessageToolCall] | None = first_request.choices[0].message.tool_calls
assert tool_calls is not None, "Model did not return any tool calls"

tool_call: ChatCompletionMessageToolCall = tool_calls[0]
args = json.loads(tool_call.function.arguments)

# now we have the arguments, we can call the function ourselves

knowledge_data: dict[str, list[dict[str, int | str]]] = get_knowledge(args["question"])

# =========================
# second api call
# ========================

# this API call is to get the final response with the response type we want

second_request: ChatCompletion = ai.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        { "role": "system", "content": "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store. Always begin your answer with 'Yep' or 'Nah' first, then provide the details. Respond in json format with 'answer' and 'source' keys." },
        { "role": "user", "content": "Is there any return/refund policy? if yes, what is the policy?" },
        cast(ChatCompletionAssistantMessageParam, first_request.choices[0].message.to_dict()),
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(knowledge_data),
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

knowledge = Knowledge(**json.loads(response))
print(knowledge.answer)
print(knowledge.source)