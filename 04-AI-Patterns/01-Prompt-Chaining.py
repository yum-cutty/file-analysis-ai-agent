from datetime import datetime as dt
import os
import logging
import json
from format_output import print_box
from typing import Optional
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

class EventValidation(BaseModel):
    """
    First LLM calls: Validate if the description by user is an event.
    """

    description: str = Field(description="Raw description, rephrased by LLM to be more clear.")
    is_event: bool = Field(description="Indicates if this description represents an event.")
    confidence_score: float = Field(description="Confidence score of the event detection.", ge=0.0, le=1.0)

class EventDetails(BaseModel):
    """
    Second LLM calls: Extract event details from the description, and parse them as structured objects.
    """

    name: str = Field(description="Name/Title of this event.")
    date: str = Field(description="Date & Time of the event. Use ISO 8601 to format this value.")
    duration: float = Field(description="Duration of the event, in minutes.")
    participants: list[str] = Field(description="List of participants attending the event.")

class EventConfirmation(BaseModel):
    """
    Third LLM calls: Confirmation message of this event that has been scheduled
    """

    confirmation_message: str = Field(description="A message confirming the event has been scheduled, written by LLM")
    calendar_link: Optional[str] = Field(description="A link to the created calendar event.") 

# ################################################################ #
#                            FUNCTIONS                             #
# ################################################################ #

# First LLM Call
def validate_event_description(user_description: str) -> EventValidation:
    """
    First Function for LLM Call: Validate if the description by user is an event.
    
    :param user_description: Description provided by user
    :type user_description: str
    :return: Validation result indicating if it's an event
    :rtype: EventValidation
    """

    logger.info("Start validating event description.")
    logger.debug(f"User description: {user_description}")

    # API call to validate event description
    try:
        request: ChatCompletion = ai.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    'role': 'system', 
                    'content': f'''
                        Today is {dt.now().strftime('%Y-%m-%d')}. 
                        Analyze if the text given describes a calendar event.

                        Respond in JSON with this schema: 
                        {{
                            "description": "Rephrase the description clearly.",
                            "is_event": "A boolean value (true or false) that indicates whether it is an event.",
                            "confidence_score": "A float between 0 and 1 that indicates the model's confidence level in this description being a calendar event."
                        }}
                    '''
                },
                {'role': 'user', 'content': user_description},
            ],
            max_tokens=4096,
            temperature=0.7,
            reasoning_effort="medium",     
            response_format={"type": "json_object"}   
        )
    except Exception as err:
        logger.error(f"Error during API call: {err}")
        raise

    response: str | None = request.choices[0].message.content
    if response is None:
        raise ValueError("No response from LLM.")

    result: EventValidation = EventValidation(**json.loads(response))

    logger.info("Event description validated successfully.")
    logger.debug(f"Validation result: Is Calendar Event - {result.is_event} with confidence score of {result.confidence_score}")
    
    return result

# Second LLM Call
def extract_event(description: str) -> EventDetails:
    """
    Second Function for LLM Call: Extract event details from the description, and parse them as structured objects.
    
    :param description: Description that was from user, and has been rephrased by LLM
    :type description: str
    :return: Extracted event details
    :rtype: EventDetails
    """
    
    logger.info("Start extracting and parsing event details.")

    # API call to extract the event details
    try:
        request: ChatCompletion = ai.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    'role': 'system',
                    'content': f'''
                        Today is {dt.now().strftime('%Y-%m-%d')}. 
                        Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference.

                        Respond in JSON with this schema:
                        {{
                            "name": "Name/Title of this event.",
                            "date": "Date & Time of the event. Use ISO 8601 to format this value.",
                            "duration": "Duration of the event, in minutes.",
                            "participants": "List of participants (names) attending the event."
                        }}
                    '''
                },
                {'role': 'user', 'content': description},
            ],
            max_tokens=4096,
            temperature=0.7,
            reasoning_effort="medium",
            response_format={"type": "json_object"}
        )
    except Exception as err:
        logger.error(f"Error during API call: {err}")
        raise

    response: str | None = request.choices[0].message.content
    if response is None:
        raise ValueError("No response from LLM.")

    result: EventDetails = EventDetails(**json.loads(response))

    logger.info("Event details extracted and parsed successfully.")
    logger.debug(f"Extracted Event Details: Name - {result.name}, Date - {result.date}, Duration - {result.duration}, Participants - {result.participants}")

    return result

# Third LLM Call
def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """
    Third Function for LLM Call: Confirmation message of this event that has been scheduled.
    
    :param event_details: The extracted event details
    :type event_details: EventDetails
    :return: Generated event confirmation message
    :rtype: EventConfirmation
    """
    
    logger.info("Start generating event confirmation message.")

    # API call to generate confirmation message
    try:
        request: ChatCompletion = ai.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    'role': 'system',
                    'content': f'''
                        Generate a natural confirmation message for the event. Sign of with your name; AI Assistant.

                        Respond in JSON with this schema:
                        {{
                            "confirmation_message": "A message confirming the event has been scheduled.",
                            "calendar_link": "An optional link (if available) to the created calendar event."
                        }}
                    '''
                },
                {'role': 'user', 'content': str(event_details)}
            ],
            max_tokens=4096,
            temperature=0.7,
            reasoning_effort="medium",
            response_format={"type": "json_object"}
        )
    except Exception as err:
        logger.error(f"Error during API call: {err}")
        raise

    response: str | None = request.choices[0].message.content
    if response is None:
        raise ValueError("No response from LLM.")

    result: EventConfirmation = EventConfirmation(**json.loads(response))

    logger.info("Event confirmation message generated successfully.")
    logger.debug(f"Confirmation Message: {result.confirmation_message}, Calendar Link: {result.calendar_link if result.calendar_link else 'N/A'}")

    return result

# ################################################################ #
#                             CHAINING                             #
# ################################################################ #

def process_calender_request(user_input: str) -> Optional[EventConfirmation]:
    logger.info("Processing calendar request.")
    logger.debug(f"User input: {user_input}")

    # 1. First LLM Call: Validate event description
    result: EventValidation = validate_event_description(user_input)

    # 2. verify it
    if (
        not result.is_event or 
        result.confidence_score < 0.7
    ):
        logger.warning(f"With description being marked as {result.is_event}, and its confidence score of marking such result is {result.confidence_score}")
        logger.warning("The provided description is not recognized as a valid event.")

        # early return if not an event
        # no further processing needed
        return None
    
    logger.info("Description validated as an event. Proceeding to extract details.")

    # 3. Second & Third LLM Call: Extract and confirm event
    confirmation: EventConfirmation = generate_confirmation(
        extract_event(result.description)
    )

    logger.info("Calendar request processed successfully.")
    return confirmation

# ################################################################ #
#                       TEST #1: Valid Input                       #
# ################################################################ #

user_input_1: str = "Set up a meeting with the design team next Tuesday at 3 PM for 1 hour to discuss the new app UI with Alice, Bob, Charlie, Eve and James."

result: Optional[EventConfirmation] = process_calender_request(user_input_1)

if result:
    lines: list[str] = [f"Message: {result.confirmation_message}"]

    if result.calendar_link:
        lines.append(f"Calendar Link: {result.calendar_link}")

    print_box("✅ EVENT SCHEDULED SUCCESSFULLY", lines)
else:
    print_box("❌ EVENT SCHEDULING FAILED", ["Not recognized as a valid event."])

# ################################################################ #
#                       TEST #2: Invalid Input                     #
# ################################################################ #

user_input_2: str = "Can you send an email to Alice and Bob to discuss the project roadmap?"

result = process_calender_request(user_input_2)
if result:
    lines: list[str] = [f"Message: {result.confirmation_message}"]
    if result.calendar_link:
        lines.append(f"Calendar Link: {result.calendar_link}")
    print_box("✅ EVENT SCHEDULED SUCCESSFULLY", lines)
else:
    print_box("❌ EVENT SCHEDULING FAILED", ["Not recognized as a valid event."])
