from asyncio import sleep
import pytest
import pytest_asyncio
from monocle_test_tools import TraceAssertion
from llamaindex_travel_agent import setup_agents, generate_session_id

supervisor = None
session_id = None

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_supervior():
    """Set up the travel booking supervisor agent."""
    global supervisor
    supervisor = await setup_agents()
    global session_id
    if session_id is None:
        session_id = generate_session_id()

@pytest.mark.asyncio
async def test_sentiment_bias_evaluation(monocle_trace_asserter):
    """v0: Basic sentiment, bias evaluation on trace - only specify eval name and expected value."""
    await monocle_trace_asserter.run_agent_async(supervisor, "llamaindex",
                        "Book a flight from San Jose to Seattle for 27th Nov 2025.")
    # Fact is implicit (trace), only specify eval template name and expected value
    monocle_trace_asserter.with_evaluation("okahu").check_eval("sentiment", "positive")\
        .check_eval("bias", "unbiased")

@pytest.mark.asyncio
async def test_quality_evaluation(monocle_trace_asserter):
    """Demonstrates using multiple evaluators (okahu and bert_score) within a single test."""
    await monocle_trace_asserter.run_agent_async(supervisor, "llamaindex",
                        "Please Book a flight from New York to Hamburg for 1st Dec 2025. Book a flight from Hamburg to Paris on January 1st. " \
                        "Then book a hotel room in Paris for 5th Jan 2026.")
    
    # Use okahu evaluator for quality metrics
    # You can chain multiple check_eval calls for different eval templates
    monocle_trace_asserter.with_evaluation("okahu").check_eval("frustration", "ok")\
        .check_eval("hallucination", "no_hallucination")
    
    # Once declared, the evaluator persists for subsequent assertions
    monocle_trace_asserter.with_evaluation("okahu").check_eval("contextual_precision", "high_precision")

@pytest.mark.asyncio
async def test_tool_agent_invocation1(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(supervisor, "llamaindex", 
                        "Book a flight from San Francisco to Mumbai for 26th April 2026. Book a two queen room at Marriott Intercontinental at Central Mumbai for 27th April 2026 for 4 nights.")
    
    monocle_trace_asserter.called_tool("lmx_book_flight_tool","lmx_flight_booking_agent") \
        .contains_input("Mumbai").contains_input("San Francisco") \
        .contains_output("San Francisco to Mumbai").contains_output("success")
    
    monocle_trace_asserter.called_tool("lmx_book_hotel_tool","lmx_hotel_booking_agent") \
        .contains_input("Central Mumbai").contains_input("Marriott Intercontinental") \
        .contains_output("booked") \
        .contains_output("Successfully booked a stay at Marriott Intercontinental in Central Mumbai") \
        .contains_output("success")
    
    monocle_trace_asserter.called_agent("lmx_flight_booking_agent")

    # example error case: check_eval will return non_toxic. Test will fail as expected since we are checking for toxic. 
    # This is to demonstrate how to use check_eval for error cases as well.
    monocle_trace_asserter.with_evaluation("okahu").check_eval("toxicity", "toxic")   

@pytest.mark.asyncio
async def test_multiple_evaluators_evaluation(monocle_trace_asserter):
    """Demonstrates using multiple evaluators (okahu and bert_score) within a single test."""
    await monocle_trace_asserter.run_agent_async(supervisor, "llamaindex",
                        "Please Book a flight from New York to Hamburg for 1st Dec 2025. Book a flight from Hamburg to Paris on January 1st. " \
                        "Then book a hotel room in Paris for 5th Jan 2026.")
    
    # Use okahu evaluator for quality metrics
    # You can chain multiple check_eval calls for different eval templates
    monocle_trace_asserter.with_evaluation("okahu").check_eval("frustration", "ok")\
        .check_eval("hallucination", "no_hallucination")
    
    # Switch to bert_score evaluator by passing options as a dictionary
    # This is an example of how you can use multiple evalauators in a single test. 
    monocle_trace_asserter.with_evaluation("bert_score", {"model_type": "bert-base-uncased"})
    
    # Switch back to okahu evaluator for additional checks
    # Once declared, the evaluator persists for subsequent assertions
    monocle_trace_asserter.with_evaluation("okahu").check_eval("contextual_precision", "high_precision")


@pytest.mark.asyncio
async def test_agent_and_tool_invocation(monocle_trace_asserter):
    await monocle_trace_asserter.run_agent_async(supervisor, "llamaindex",
                    "Book a flight from San Francisco to Mumbai on April 30th 2026. Book hotel Marriott in Central Mumbai. Also how is the weather going to be in Mumbai?", session_id=session_id)

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