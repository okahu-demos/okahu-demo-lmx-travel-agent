from asyncio import sleep
import pytest
import pytest_asyncio
from monocle_test_tools import TraceAssertion
from llamaindex_travel_agent import setup_agents

supervisor = None

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_supervior():
    """Set up the travel booking supervisor agent."""
    global supervisor
    supervisor = await setup_agents()

@pytest.mark.asyncio
async def test_agent_and_tool_invocation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(supervisor, "llamaindex",
                    "Book a flight from San Francisco to Mumbai on April 30th 2026. Book hotel Marriott in Central Mumbai. Also how is the weather going to be in Mumbai?")

    monocle_trace_asserter.called_tool("lmx_book_flight_tool","lmx_flight_booking_agent") \
        .contains_input("Mumbai").contains_input("San Francisco") \
        .contains_output("Successfully booked a flight from San Francisco to Mumbai").contains_output("booked")
 
    monocle_trace_asserter.called_tool("lmx_book_hotel_tool","lmx_hotel_booking_agent") \
        .contains_input("Marriott").contains_input("Mumbai") \
        .contains_output("Marriott") \
        .contains_output("Central Mumbai") \
        .contains_output("booked")
 
    monocle_trace_asserter.called_tool("demo_get_weather","lmx_weather_agent") \
        .contains_input("city").contains_input("Mumbai") \
        .contains_output("temperature")
    
    monocle_trace_asserter.called_agent("lmx_weather_agent") \
        .contains_output("The weather in Mumbai") \
        .contains_output("weather") \
        .contains_output("Mumbai")
 
    monocle_trace_asserter.called_agent("lmx_hotel_booking_agent") \
        .contains_output("Marriott") \
        .contains_output("Central Mumbai") \
        .contains_output("successfully") \
        .contains_output("booked")
    
    monocle_trace_asserter.called_agent("lmx_flight_booking_agent") \
        .contains_output("San Francisco to Mumbai") \
        .contains_output("successfully") \
        .contains_output("booked")
    
if __name__ == "__main__":
    pytest.main([__file__]) 