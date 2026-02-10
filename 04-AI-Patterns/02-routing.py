from datetime import datetime as dt
import json
import logging
import os
from format_output import print_box
from typing import Literal, Optional
from dotenv import load_dotenv
from groq import BaseModel, Groq
from groq.types.chat import ChatCompletion
from pydantic import Field

# ################################################################ #
#                          SETUP LOGGING                           #
# ################################################################ #

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s - %(message)s", # TIME - SEVERITY - MESSAGE
    datefmt="%Y-%m-%d %H:%M:%S",
)

# the name (label) of this logger
# this is just for best practices
# `__name__` holds the name of the current/parent module
logger: logging.Logger = logging.getLogger(__name__)

# ################################################################ #
#                             LOAD ENV                             #
# ################################################################ #

load_dotenv()
ai: Groq = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ################################################################ #
#                              PARAMS                              #
# ################################################################ #

class RequestType(BaseModel):
    """
    First LLM Calls: Classify the type of calendar event request.
    """

    description: str = Field(description="Raw description, rephrased by LLM to be more clear.")
    request_type: Literal["new_event", "modify_event", "other"] = Field(description="Type of calender event request being made")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence score of the request type.")

class NewEventDetails(BaseModel):
    """
    Second LLM Calls: Extract details - New Event.
    """

    name: str = Field(description="Name of the new event.")
    date: str = Field(description="Date & Time of the new event, in ISO 8601 format.")
    duration: float = Field(description="Duration of the new event, in minutes.")
    participants: list[str] = Field(description="List of participants' names for the new event.")

class ModifyEventDetails(BaseModel):
    """
    Second LLM Calls: Extract details - Modify Event.
    """

    description: str = Field(description="Changes description for the event modification.")
    updated_date: str = Field(description="Updated Date & Time of the event, in ISO 8601 format.")
    participants_to_add: list[str] = Field(description="List of participants' names to add to the event.")
    participants_to_remove: list[str] = Field(description="List of participants' names to remove from the event.")

class ModifyConfirmation(BaseModel):
    """
    Third LLM Calls: Confirm modification details.
    """

    success: bool = Field(description="Whether the operation was successful.")
    message: str = Field(description="User-Friendly response message.")

# ################################################################ #
#                            FUNCTIONS                             #
# ################################################################ #

def clasify_request(description: str) -> RequestType:
    """
    First Function for LLM Calls: Classify the type of calendar event request.
    
    :param description: Raw description of the calendar event request by the user.
    :type description: str
    :return: Classified request type with rephrased description and confidence score.
    :rtype: RequestType
    """

    logger.info("Classifying request type...")
    logger.debug(f"Input description: {description}")

    # API call to classify the request type
    try:
        request: ChatCompletion = ai.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    'role': 'system', 
                    'content': '''
                        Determine if this is a request to create a new calendar event or modify an existing one.

                        Respond in JSON format with this schema:
                        {
                            "description": "rephrased the description clearly",
                            "request_type": "one of the Literal type based on the description: new_event, modify_event, other",
                            "confidence_score": "A float between 0 and 1 that indicates the model's confidence level in this description being a calendar event."
                        }
                    '''
                },
                {'role': 'user', 'content': description}
            ],
            max_tokens=4096,
            temperature=0.7,
            reasoning_effort="medium",     
            response_format={"type": "json_object"}             
        )
    except Exception as err:
        logger.error(f"Error during request classification: {err}")
        raise

    response: str | None = request.choices[0].message.content
    if response is None:
        raise ValueError("No response from LLM.")

    result: RequestType = RequestType(**json.loads(response))

    logger.info("Request classified successfully.")
    logger.debug(f"Classification result: The modified description is '{result.description}', request type is '{result.request_type}' with confidence score of {result.confidence_score}")

    return result

def handle_new_event(description: str) -> ModifyConfirmation:
    """
    Second Function for LLM Calls: Extract details - New Event.
        
    :param description: Description
    :type description: str
    :return: Description
    :rtype: ModifyConfirmation
    """

    logger.info("Handling new event details extraction...")

    # API call to extract new event details
    try:
        request: ChatCompletion = ai.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    'role': 'system',
                    'content': f'''
                        Today is {dt.now().strftime('%Y-%m-%d')}.
                        Extract details for creating a new calendar event.

                        Respond in JSON format with this schema:
                        {{
                            "name": "Name of the new event.",
                            "date": "Date & Time of the new event, in ISO 8601 format.",
                            "duration": "Duration of the new event, in minutes.",
                            "participants": "List of participants' names for the new event."
                        }}
                    '''
                },
                {'role': 'user', 'content': description}
            ],
            max_tokens=4096,
            temperature=0.7,
            reasoning_effort="medium",     
            response_format={"type": "json_object"}
        )
    except Exception as err:
        logger.error(f"Error during new event details extraction: {err}")
        raise

    response: str | None = request.choices[0].message.content
    if response is None:
        raise ValueError("No response from LLM.")

    result: NewEventDetails = NewEventDetails(**json.loads(response))

    logger.info("New event details extracted successfully.")
    logger.debug(f"New event details: The event '{result.name}' is scheduled on {result.date} for {result.duration} minutes with participants {result.participants}")

    return ModifyConfirmation(
        success=True,
        message=f"New event created called: '{result.name}', scheduled on {result.date} for {result.duration} minutes with participants {', '.join(result.participants)}.",
    )

def handle_modify_event(description: str) -> ModifyConfirmation:
    """
    Docstring for handle_modify_event
    
    :param description: Description
    :type description: str
    :return: Description
    :rtype: ModifyConfirmation
    """

    logger.info("Handling modify event details extraction...")

    # API call to extract modify event details
    try:
        request: ChatCompletion = ai.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    'role': 'system',
                    'content': f'''
                        Today is {dt.now().strftime('%Y-%m-%d')}.
                        Extract details for modifying an existing calendar event.

                        Respond in JSON format with this schema:
                        {{
                            "description": "Changes description for the event modification.",
                            "updated_date": "Updated Date & Time of the event, in ISO 8601 format.",
                            "participants_to_add": "List of participants' names to add to the event.",
                            "participants_to_remove": "List of participants' names to remove from the event."
                        }}
                    '''
                },
                {'role': 'user', 'content': description}
            ],
            max_tokens=4096,
            temperature=0.7,
            reasoning_effort="medium",     
            response_format={"type": "json_object"}
        )
    except Exception as err:
        logger.error(f"Error during modify event details extraction: {err}")
        raise

    response: str | None = request.choices[0].message.content
    if response is None:
        raise ValueError("No response from LLM.")
    
    result: ModifyEventDetails = ModifyEventDetails(**json.loads(response))

    logger.info("Modify event details extracted successfully.")
    logger.debug(f"Modify event details: The event modification description is '{result.description}', updated date is {result.updated_date}, participants to add: {result.participants_to_add}, participants to remove: {result.participants_to_remove}")

    return ModifyConfirmation(
        success=True,
        message=f"Event modified with description: '{result.description}', updated date: {result.updated_date}, participants to add: {', '.join(result.participants_to_add)}, participants to remove: {', '.join(result.participants_to_remove)}."
    )

# ################################################################ #
#                             ROUTING                              #
# ################################################################ #

def process_calendar_request(description: str) -> Optional[ModifyConfirmation]:

    logger.info("Processing calendar request...")

    # 1. First LLM Call: Classify the request type
    classification: RequestType = clasify_request(description)

    # 2. Check Confidence Score
    if classification.confidence_score < 0.7:
        logger.warning(f"With low confidence score of {classification.confidence_score}, unable to process the request.")
        return None
    
    # 3. Route to appropriate handler based on request type
    if classification.request_type == "new_event":
        return handle_new_event(classification.description)
    elif classification.request_type == "modify_event":
        return handle_modify_event(classification.description)
    else:
        logger.info("Request type is 'other'; no action taken.")
        return None
    
# ################################################################ #
#                        TEST #1: NEW EVENT                        #
# ################################################################ #

new_event_input: str = "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob"
result: Optional[ModifyConfirmation] = process_calendar_request(new_event_input)

if result:
    logger.info(f"Test #1 Result: {result.message}")
    lines: list[str] = [f"Test #1 Result: {result.message}"]
    print_box("✅ EVENT CREATED SUCCESSFULLY", lines)
else:
    logger.info("Test #1 Result: Unable to process the request.")
    lines: list[str] = ["Unable to process the request."]
    print_box("❌ EVENT CREATION FAILED", lines)

# ################################################################ #
#                      TEST #2: MODIFY EVENT                       #
# ################################################################ #

modify_event_input: str = "Can you move the team meeting with Alice and Bob to Wednesday at 3pm instead?"
result = process_calendar_request(modify_event_input)

if result:
    logger.info(f"Test #2 Result: {result.message}")
    lines = [f"Test #2 Result: {result.message}"]
    print_box("✅ EVENT MODIFIED SUCCESSFULLY", lines)
else:
    logger.info("Test #2 Result: Unable to process the request.")
    lines = ["Unable to process the request."]
    print_box("❌ EVENT MODIFICATION FAILED", lines)

# ################################################################ #
#                      TEST #3: INVALID INPUT                      #
# ################################################################ #

invalid_input: str = "What's the weather like today?"
result = process_calendar_request(invalid_input)

if result:
    logger.info(f"Test #3 Result: {result.message}")
    lines = [f"Test #3 Result: {result.message}"]
    print_box("✅ REQUEST PROCESSED SUCCESSFULLY", lines)
else:
    logger.info("Test #3 Result: Unable to process the request.")
    lines = ["Unable to process the request."]
    print_box("❌ REQUEST PROCESSING FAILED", lines)