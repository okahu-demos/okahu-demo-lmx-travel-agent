from asyncio import sleep
import logging
import pytest 
import logging
import os
from dotenv import load_dotenv

from llamaindex_travel_agent import (
    setup_agents
)
from monocle_test_tools import TestCase, MonocleValidator

OKAHU_API_KEY = os.environ.get('OKAHU_API_KEY')
logging.basicConfig(level=logging.WARN)
load_dotenv()


agent_test_cases:list[TestCase] = [
    {
        "test_input": ["Book a flight from San Jose to Boston and a book hotel stay at Hyatt Hotel, and tell the weather in Boston."],
        "test_output": "Here are your booking details and weather information for Boston:\n\n- **Flight**: Successfully booked fro...n.\n- **Weather in Boston**: The current temperature is 69°F.",
        "comparer": "similarity",
    },
    {
        "test_input": ["Book a flight from San Francisco to Mumbai. Book a two queen room at Marriot Intercontinental at Central Mumbai and tell me the weather of Mumbai."],
        "test_spans": [

            {
            "span_type": "agentic.invocation",
            "entities": [
                {"type": "agent", "name": "lmx_coordinator"}
                ]
            },
            {
            "span_type": "agentic.invocation",
            "entities": [
                {"type": "agent", "name": "lmx_flight_booking_agent"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "lmx_book_flight_tool"},
                    {"type": "agent", "name": "lmx_flight_booking_agent"}
                ]
            },
            {
            "span_type": "agentic.invocation",
            "entities": [
                {"type": "agent", "name": "lmx_hotel_booking_agent"}
                ]
            },
            {
                "span_type": "agentic.tool.invocation",
                "entities": [
                    {"type": "tool", "name": "lmx_book_hotel_tool"},
                     {"type": "agent", "name": "lmx_hotel_booking_agent"}
                ]
            }
        ]
    }
]

@MonocleValidator().monocle_testcase(agent_test_cases)
async def test_run_agents(my_test_case: TestCase):
    agent_workflow = await setup_agents()
    await MonocleValidator().test_agent_async(agent_workflow, "llamaindex", my_test_case)
    await sleep(10)


if __name__ == "__main__":
    pytest.main([__file__]) 