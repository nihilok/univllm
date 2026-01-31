"""
Example usage of MCP tool calling with univllm.

This demonstrates how to use the MCP-compatible tool calling API
with different LLM providers.
"""

import asyncio
import json
from univllm import UniversalLLMClient, ToolDefinition


# Define tools using MCP format
def get_weather_tool():
    """Example weather tool definition."""
    return ToolDefinition(
        name="get_weather",
        description="Get current weather information for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or zip code"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        }
    )


def calculate_tool():
    """Example calculator tool definition."""
    return ToolDefinition(
        name="calculate",
        description="Perform basic arithmetic calculations",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        }
    )


def search_tool():
    """Example search tool definition."""
    return ToolDefinition(
        name="search_web",
        description="Search the web for information",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    )


# Simulate tool execution
def execute_tool(tool_name: str, arguments: dict) -> str:
    """
    Simulate executing a tool call.
    In a real MCP implementation, this would invoke the actual tool.
    """
    if tool_name == "get_weather":
        location = arguments.get("location")
        units = arguments.get("units", "fahrenheit")
        # Simulate weather API response
        return f"Current weather in {location}: 72°{units[0].upper()}, partly cloudy"
    
    elif tool_name == "calculate":
        expression = arguments.get("expression")
        try:
            # NOTE: In production, use a safe expression evaluator like simpleeval
            # or ast.literal_eval for simple cases. Never use eval() with user input!
            # This is for demonstration purposes only.
            # Example safe alternative: from simpleeval import simple_eval
            # result = simple_eval(expression)
            
            # For this example, we'll use a restricted approach
            # Only allow basic arithmetic with numbers
            import re
            if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
                result = eval(expression)  # Still not ideal, but safer with validation
                return f"Result: {result}"
            else:
                return "Error: Expression contains disallowed characters"
        except Exception as e:
            return f"Error calculating: {e}"
    
    elif tool_name == "search_web":
        query = arguments.get("query")
        max_results = arguments.get("max_results", 5)
        # Simulate search results
        return f"Found {max_results} results for '{query}'"
    
    else:
        return f"Unknown tool: {tool_name}"


async def basic_tool_calling_example():
    """Basic example of tool calling with a single tool."""
    print("\n=== Basic Tool Calling Example ===\n")
    
    client = UniversalLLMClient()
    
    # Define available tools
    tools = [get_weather_tool()]
    
    # Initial user request
    messages = [
        {"role": "user", "content": "What's the weather in New York?"}
    ]
    
    print("User: What's the weather in New York?")
    
    # First API call - LLM decides to use a tool
    response = await client.complete(
        messages=messages,
        model="gpt-4o",
        tools=tools,
        tool_choice="auto"
    )
    
    # Check if the model wants to call a tool
    if response.tool_calls:
        print(f"\nModel wants to call tool: {response.tool_calls[0].name}")
        print(f"Arguments: {json.dumps(response.tool_calls[0].arguments, indent=2)}")
        
        # Execute the tool
        tool_result = execute_tool(
            response.tool_calls[0].name,
            response.tool_calls[0].arguments
        )
        print(f"\nTool result: {tool_result}")
        
        # Add assistant's tool call and tool result to conversation
        messages.append({
            "role": "assistant",
            "content": f"I need to check the weather. Calling {response.tool_calls[0].name}..."
        })
        messages.append({
            "role": "user",
            "content": f"Tool result: {tool_result}"
        })
        
        # Second API call - LLM uses tool result to generate final response
        final_response = await client.complete(
            messages=messages,
            model="gpt-4o"
        )
        
        print(f"\nAssistant: {final_response.content}")
    else:
        print(f"\nAssistant: {response.content}")


async def multiple_tools_example():
    """Example with multiple tools available."""
    print("\n=== Multiple Tools Example ===\n")
    
    client = UniversalLLMClient()
    
    # Define multiple available tools
    tools = [
        get_weather_tool(),
        calculate_tool(),
        search_tool()
    ]
    
    messages = [
        {"role": "user", "content": "Calculate 15 * 23 and tell me the weather in London"}
    ]
    
    print("User: Calculate 15 * 23 and tell me the weather in London")
    
    response = await client.complete(
        messages=messages,
        model="gpt-4o",
        tools=tools,
        tool_choice="auto"
    )
    
    if response.tool_calls:
        print(f"\nModel wants to call {len(response.tool_calls)} tool(s):")
        
        for tool_call in response.tool_calls:
            print(f"\n- {tool_call.name}")
            print(f"  Arguments: {json.dumps(tool_call.arguments, indent=4)}")
            
            # Execute each tool
            result = execute_tool(tool_call.name, tool_call.arguments)
            print(f"  Result: {result}")


async def anthropic_tool_example():
    """Example using Anthropic/Claude with tools."""
    print("\n=== Anthropic Tool Calling Example ===\n")
    
    client = UniversalLLMClient()
    
    tools = [calculate_tool()]
    
    messages = [
        {"role": "user", "content": "What is 42 divided by 7?"}
    ]
    
    print("User: What is 42 divided by 7?")
    
    response = await client.complete(
        messages=messages,
        model="claude-sonnet-4-20250514",
        tools=tools
    )
    
    if response.tool_calls:
        print(f"\nClaude wants to use: {response.tool_calls[0].name}")
        print(f"Arguments: {json.dumps(response.tool_calls[0].arguments, indent=2)}")
        
        # Execute the tool
        result = execute_tool(
            response.tool_calls[0].name,
            response.tool_calls[0].arguments
        )
        print(f"\nTool result: {result}")
        print(f"\nClaude's message: {response.content}")


async def tool_choice_example():
    """Example demonstrating different tool_choice options."""
    print("\n=== Tool Choice Options Example ===\n")
    
    client = UniversalLLMClient()
    tools = [get_weather_tool(), calculate_tool()]
    
    # Force no tool use
    print("1. tool_choice='none' - Force the model to NOT use tools:")
    response = await client.complete(
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        model="gpt-4o",
        tools=tools,
        tool_choice="none"
    )
    print(f"   Response: {response.content}")
    print(f"   Tool calls: {response.tool_calls}")
    
    # Let model decide (auto)
    print("\n2. tool_choice='auto' - Let the model decide:")
    response = await client.complete(
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        model="gpt-4o",
        tools=tools,
        tool_choice="auto"
    )
    if response.tool_calls:
        print(f"   Model chose to use: {response.tool_calls[0].name}")
    else:
        print(f"   Model chose not to use tools")


async def dict_format_example():
    """Example using dictionary format for tools (MCP compatible)."""
    print("\n=== Dictionary Format Example ===\n")
    
    client = UniversalLLMClient()
    
    # Define tools as dictionaries (MCP format)
    tools = [
        {
            "name": "get_current_time",
            "description": "Get the current time in a specific timezone",
            "input_schema": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone name (e.g., 'America/New_York')"
                    }
                },
                "required": ["timezone"]
            }
        }
    ]
    
    response = await client.complete(
        messages=[{"role": "user", "content": "What time is it in Tokyo?"}],
        model="gpt-4o",
        tools=tools  # Pass tools as dictionaries
    )
    
    if response.tool_calls:
        print(f"Tool called: {response.tool_calls[0].name}")
        print(f"Arguments: {json.dumps(response.tool_calls[0].arguments, indent=2)}")


async def main():
    """Run all examples."""
    print("=" * 60)
    print("MCP Tool Calling Examples for univllm")
    print("=" * 60)
    
    # Note: These examples demonstrate the API usage but require valid API keys
    # Uncomment the examples you want to run:
    
    # await basic_tool_calling_example()
    # await multiple_tools_example()
    # await anthropic_tool_example()
    # await tool_choice_example()
    # await dict_format_example()
    
    print("\n" + "=" * 60)
    print("Note: Set OPENAI_API_KEY and ANTHROPIC_API_KEY environment")
    print("variables to run these examples with real API calls.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
