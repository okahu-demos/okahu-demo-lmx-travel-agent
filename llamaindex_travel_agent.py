
import asyncio
import time
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.llms.openai import OpenAI
from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry
from llama_index.core.agent import ReActAgent
from llama_index.tools.mcp import aget_tools_from_mcp_url
import logging

logger = logging.getLogger(__name__)

setup_monocle_telemetry(workflow_name="okahu_demos_llamaindex_travel_agent", monocle_exporters_list='okahu,file')

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

async def setup_agents():
    
    async def get_mcp_tools():
        """Get tools from the MCP weather server."""
        try:
            weather_tools = await aget_tools_from_mcp_url(
                "http://localhost:8001/weather/mcp/"
            )
            return weather_tools
        except Exception as e:
            logger.warning(f"Failed to get MCP tools: {e}")
            return []

    # Get MCP weather tools
    weather_tools = await get_mcp_tools()
   
    llm = OpenAI(model="gpt-4o", additional_kwargs={"stream_options": {"include_usage": True}})

    flight_tool = FunctionTool.from_defaults(
        fn=book_flight,
        name="lmx_book_flight_tool",
        description="Books a flight from one airport to another."
    )
    flight_agent = FunctionAgent(
                                name="lmx_flight_booking_agent",
                                tools=[flight_tool],
                                llm=llm,
                                system_prompt="""You are a flight booking agent who books flights as per the request. 
                                        When you receive a flight booking request, immediately use the book_flight tool to complete the booking.
                                        After successfully booking the flight, you MUST handoff back to lmx_coordinator with the booking confirmation message.""",
                                description="Flight booking agent",
                                can_handoff_to=["lmx_coordinator"]
                            )

    hotel_tool = FunctionTool.from_defaults(
        fn=book_hotel,
        name="lmx_book_hotel_tool",
        description="Books a hotel stay."
    )
    hotel_agent = FunctionAgent(
                                name="lmx_hotel_booking_agent",
                                tools=[hotel_tool],
                                llm=llm,
                                system_prompt="""You are a hotel booking agent who books hotels as per the request.
                                        When you receive a hotel booking request, immediately use the book_hotel tool to complete the booking.
                                        After successfully booking the hotel, you MUST handoff back to lmx_coordinator with the booking confirmation message.""",
                                description="Hotel booking agent",
                                can_handoff_to=["lmx_coordinator"]
                            )
    
    weather_agent = FunctionAgent(
                                name="lmx_weather_agent",
                                tools=[weather_tool for weather_tool in weather_tools],
                                llm=llm,
                                system_prompt="""You are a weather agent who provides weather information as per the request.
                                        When you receive a weather information request, immediately use the weather tools to provide the information.
                                        After successfully providing the weather information, you MUST handoff back to lmx_coordinator with the weather details.""",
                                description="Weather information agent",
                                can_handoff_to=["lmx_coordinator"]
                            )

    coordinator = FunctionAgent(
                                name="lmx_coordinator",
                                llm=llm,
                                system_prompt=
                                """You are a coordinator agent who manages flight and hotel booking agents. 
                         
                                    For each user request:
                                    1. First delegate flight booking to the lmx_flight_booking_agent agent
                                    2. After flight booking is complete, delegate hotel booking to the lmx_hotel_booking_agent agent  
                                    3. Once both bookings are complete, use weather tools if weather information is requested
                                    4. Provide a consolidated response with all booking details and weather information
                                    
                                    Always ensure both agents complete their tasks and gather all information before providing the final response.
                                    Continue delegating until all tasks are done.""",
                                description="Travel booking coordinator agent",
                                can_handoff_to=["lmx_flight_booking_agent", "lmx_hotel_booking_agent", "lmx_weather_agent"])

    agent_workflow = AgentWorkflow(
        handoff_prompt="""As soon as you have figured out the requirements, 
        If you need to delegate the task to another agent, then delegate the task to that agent.
        For eg if you need to book a flight, then delegate the task to flight agent.
        If you need to book a hotel, then delegate the task to hotel agent.
        If you can book hotel or flight direclty, then do that and collect the response and then handoff to the supervisor agent.
        If you need to provide weather information, then delegate the task to weather agent.
        {agent_info}
        """,
        agents=[coordinator, flight_agent, hotel_agent, weather_agent],
        root_agent=coordinator.name
    )
    return agent_workflow

async def run_agent(user_msg: str = None):
    """Test multi-agent interaction with flight and hotel booking."""

    agent_workflow = await setup_agents()
    
    # If no user_msg provided, use default requests (for manual testing)
    if user_msg is None:
        requests = [
            "Book a flight from San Jose to Boston and a book hotel stay at Hyatt Hotel, and tell the weather in Boston.",
            ]
        for req in requests:
            resp = await agent_workflow.run(user_msg=req)
            print(resp)
        return None
    else:
        # For test framework: process single message and return response as string
        resp = await agent_workflow.run(user_msg=user_msg)
        # Extract the string response from the workflow result
        if hasattr(resp, 'response'):
            return str(resp.response)
        elif isinstance(resp, dict) and 'response' in resp:
            return str(resp['response'])
        else:
            return str(resp)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    asyncio.run(run_agent())
    
