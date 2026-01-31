"""Tests for MCP tool calling support.

These tests mock MCP server I/O to validate the tool calling implementation
follows the MCP (Model Context Protocol) format.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from univllm import UniversalLLMClient, ProviderType, ToolDefinition, ToolCall


@pytest.fixture
def weather_tool():
    """Sample weather tool definition in MCP format."""
    return ToolDefinition(
        name="get_weather",
        description="Get current weather information for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or zip code"
                }
            },
            "required": ["location"]
        }
    )


@pytest.fixture
def calculator_tool():
    """Sample calculator tool definition in MCP format."""
    return ToolDefinition(
        name="calculate",
        description="Perform basic arithmetic calculations",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    )


class TestMCPToolDefinitions:
    """Test MCP tool definition format and validation."""

    def test_tool_definition_creation(self, weather_tool):
        """Test creating a tool definition with proper MCP format."""
        assert weather_tool.name == "get_weather"
        assert weather_tool.description == "Get current weather information for a location"
        assert "type" in weather_tool.input_schema
        assert weather_tool.input_schema["type"] == "object"
        assert "properties" in weather_tool.input_schema
        assert "location" in weather_tool.input_schema["properties"]

    def test_tool_definition_from_dict(self):
        """Test creating tool definition from dictionary (MCP tools/list response format)."""
        tool_dict = {
            "name": "search",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
        tool = ToolDefinition(**tool_dict)
        assert tool.name == "search"
        assert tool.description == "Search the web"

    def test_multiple_tools_list(self, weather_tool, calculator_tool):
        """Test handling multiple tools (MCP tools/list format)."""
        tools = [weather_tool, calculator_tool]
        assert len(tools) == 2
        assert tools[0].name == "get_weather"
        assert tools[1].name == "calculate"


class TestMCPToolCalling:
    """Test MCP tool calling flow with mocked provider I/O."""

    @pytest.mark.asyncio
    async def test_openai_tool_call_format(self, weather_tool, monkeypatch):
        """Test OpenAI provider formats tools correctly for API."""
        client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="test_key")
        provider = client.provider_instance

        # Mock the OpenAI API response with tool call
        mock_message = MagicMock()
        mock_message.content = None
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = json.dumps({"location": "New York"})
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = None

        # Mock the create method
        async def mock_create(**kwargs):
            # Verify tools are formatted correctly (OpenAI format)
            if "tools" in kwargs:
                assert len(kwargs["tools"]) == 1
                tool = kwargs["tools"][0]
                assert tool["type"] == "function"
                assert "function" in tool
                assert tool["function"]["name"] == "get_weather"
                assert "parameters" in tool["function"]
            return mock_response

        monkeypatch.setattr(
            provider.client.chat.completions, "create", mock_create
        )

        # Test with tool
        response = await client.complete(
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            model="gpt-4o",
            tools=[weather_tool]
        )

        # Verify response contains tool calls
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments["location"] == "New York"
        assert response.tool_calls[0].id == "call_123"

    @pytest.mark.asyncio
    async def test_anthropic_tool_call_format(self, weather_tool, monkeypatch):
        """Test Anthropic provider formats tools correctly for API."""
        client = UniversalLLMClient(provider=ProviderType.ANTHROPIC, api_key="test_key")
        provider = client.provider_instance

        # Mock the Anthropic API response with tool use
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "I'll check the weather for you."

        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "toolu_123"
        mock_tool_block.name = "get_weather"
        mock_tool_block.input = {"location": "New York"}

        mock_response = MagicMock()
        mock_response.content = [mock_text_block, mock_tool_block]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.stop_reason = "tool_use"
        mock_response.usage = None

        # Mock the create method
        async def mock_create(**kwargs):
            # Verify tools are formatted correctly (Anthropic format)
            if "tools" in kwargs:
                assert len(kwargs["tools"]) == 1
                tool = kwargs["tools"][0]
                assert tool["name"] == "get_weather"
                assert tool["description"] == weather_tool.description
                assert "input_schema" in tool
            return mock_response

        monkeypatch.setattr(
            provider.client.messages, "create", mock_create
        )

        # Test with tool
        response = await client.complete(
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            model="claude-sonnet-4-20250514",
            tools=[weather_tool]
        )

        # Verify response contains tool calls
        assert response.content == "I'll check the weather for you."
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments["location"] == "New York"
        assert response.tool_calls[0].id == "toolu_123"

    @pytest.mark.asyncio
    async def test_tool_choice_parameter(self, weather_tool, calculator_tool, monkeypatch):
        """Test tool_choice parameter controls tool usage."""
        client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="test_key")
        provider = client.provider_instance

        captured_kwargs = {}

        async def mock_create(**kwargs):
            captured_kwargs.update(kwargs)
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Result"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4o"
            mock_response.usage = None
            return mock_response

        monkeypatch.setattr(
            provider.client.chat.completions, "create", mock_create
        )

        # Test with tool_choice="auto"
        await client.complete(
            messages=[{"role": "user", "content": "Calculate 2+2"}],
            model="gpt-4o",
            tools=[weather_tool, calculator_tool],
            tool_choice="auto"
        )
        assert captured_kwargs.get("tool_choice") == "auto"

        # Test with tool_choice="none"
        captured_kwargs.clear()
        await client.complete(
            messages=[{"role": "user", "content": "Calculate 2+2"}],
            model="gpt-4o",
            tools=[weather_tool, calculator_tool],
            tool_choice="none"
        )
        assert captured_kwargs.get("tool_choice") == "none"

    @pytest.mark.asyncio
    async def test_no_tools_backward_compatibility(self, monkeypatch):
        """Test that requests without tools still work (backward compatibility)."""
        client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="test_key")
        provider = client.provider_instance

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4o"
        mock_response.usage = None

        async def mock_create(**kwargs):
            # Verify no tools in request
            assert "tools" not in kwargs or kwargs["tools"] is None
            return mock_response

        monkeypatch.setattr(
            provider.client.chat.completions, "create", mock_create
        )

        response = await client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o"
        )

        assert response.content == "Hello, world!"
        assert response.tool_calls is None

    @pytest.mark.asyncio
    async def test_tool_call_with_dict_format(self, monkeypatch):
        """Test tools can be passed as dictionaries (MCP format)."""
        client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="test_key")
        provider = client.provider_instance

        captured_tools = None

        async def mock_create(**kwargs):
            nonlocal captured_tools
            captured_tools = kwargs.get("tools")
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message = MagicMock()
            mock_response.choices[0].message.content = "Result"
            mock_response.choices[0].message.tool_calls = None
            mock_response.choices[0].finish_reason = "stop"
            mock_response.model = "gpt-4o"
            mock_response.usage = None
            return mock_response

        monkeypatch.setattr(
            provider.client.chat.completions, "create", mock_create
        )

        # Pass tools as dictionaries (MCP format)
        tool_dict = {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }
        }

        await client.complete(
            messages=[{"role": "user", "content": "Test"}],
            model="gpt-4o",
            tools=[tool_dict]
        )

        # Verify tool was properly converted
        assert captured_tools is not None
        assert len(captured_tools) == 1
        assert captured_tools[0]["function"]["name"] == "test_tool"


class TestMCPToolCallSimulation:
    """Test simulated MCP tool call workflow."""

    @pytest.mark.asyncio
    async def test_complete_tool_call_workflow(self, weather_tool, monkeypatch):
        """Test complete workflow: user message -> tool call -> tool result -> final response."""
        client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="test_key")
        provider = client.provider_instance

        call_count = [0]

        async def mock_create(**kwargs):
            call_count[0] += 1
            
            # First call: LLM decides to use tool
            if call_count[0] == 1:
                mock_message = MagicMock()
                mock_message.content = None
                mock_tool_call = MagicMock()
                mock_tool_call.id = "call_123"
                mock_tool_call.function.name = "get_weather"
                mock_tool_call.function.arguments = json.dumps({"location": "New York"})
                mock_message.tool_calls = [mock_tool_call]

                mock_choice = MagicMock()
                mock_choice.message = mock_message
                mock_choice.finish_reason = "tool_calls"

                mock_response = MagicMock()
                mock_response.choices = [mock_choice]
                mock_response.model = "gpt-4o"
                mock_response.usage = None
                return mock_response
            
            # Second call: LLM processes tool result and gives final answer
            else:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message = MagicMock()
                mock_response.choices[0].message.content = (
                    "The weather in New York is 72°F and partly cloudy."
                )
                mock_response.choices[0].message.tool_calls = None
                mock_response.choices[0].finish_reason = "stop"
                mock_response.model = "gpt-4o"
                mock_response.usage = None
                return mock_response

        monkeypatch.setattr(
            provider.client.chat.completions, "create", mock_create
        )

        # Step 1: Initial request with tools
        response = await client.complete(
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            model="gpt-4o",
            tools=[weather_tool]
        )

        # Verify tool call was requested
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]
        assert tool_call.name == "get_weather"
        assert tool_call.arguments["location"] == "New York"

        # Step 2: Simulate executing the tool (this would be done by MCP server)
        # In real scenario, MCP server would execute: tools/call with the arguments
        tool_result = "Current weather in New York: Temperature: 72°F, Conditions: Partly cloudy"

        # Step 3: Send tool result back to LLM for final response
        # Note: In actual implementation, you'd format the tool result as a message
        # This demonstrates the workflow concept
        final_response = await client.complete(
            messages=[
                {"role": "user", "content": "What's the weather in New York?"},
                {"role": "assistant", "content": f"Tool call: {tool_call.name}"},
                {"role": "user", "content": f"Tool result: {tool_result}"}
            ],
            model="gpt-4o"
        )

        assert "72°F" in final_response.content
        assert "partly cloudy" in final_response.content.lower()

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, weather_tool, calculator_tool, monkeypatch):
        """Test handling multiple tool calls in a single response."""
        client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="test_key")
        provider = client.provider_instance

        # Mock response with multiple tool calls
        mock_message = MagicMock()
        mock_message.content = None
        
        mock_tool_call_1 = MagicMock()
        mock_tool_call_1.id = "call_1"
        mock_tool_call_1.function.name = "get_weather"
        mock_tool_call_1.function.arguments = json.dumps({"location": "New York"})
        
        mock_tool_call_2 = MagicMock()
        mock_tool_call_2.id = "call_2"
        mock_tool_call_2.function.name = "calculate"
        mock_tool_call_2.function.arguments = json.dumps({"expression": "2+2"})
        
        mock_message.tool_calls = [mock_tool_call_1, mock_tool_call_2]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = None

        async def mock_create(**kwargs):
            return mock_response

        monkeypatch.setattr(
            provider.client.chat.completions, "create", mock_create
        )

        response = await client.complete(
            messages=[{"role": "user", "content": "Weather in NY and calculate 2+2"}],
            model="gpt-4o",
            tools=[weather_tool, calculator_tool]
        )

        # Verify multiple tool calls
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[1].name == "calculate"

    @pytest.mark.asyncio
    async def test_gemini_tool_call_format(self, weather_tool, monkeypatch):
        """Test Gemini provider formats tools correctly for API."""
        client = UniversalLLMClient(provider=ProviderType.GEMINI, api_key="test_key")
        provider = client.provider_instance

        # Mock Gemini API response with function call
        mock_function_call = MagicMock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "New York"}
        mock_function_call.id = None

        mock_part = MagicMock()
        # Don't set text attribute so it's treated as function call
        del mock_part.text
        mock_part.function_call = mock_function_call
        
        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        # Mock the generate_content method
        async def mock_generate_content(**kwargs):
            # Verify tools are formatted correctly (Gemini format)
            if hasattr(kwargs.get("config"), "tools") and kwargs["config"].tools:
                tools = kwargs["config"].tools
                assert len(tools) == 1
                tool = tools[0]
                assert hasattr(tool, "function_declarations")
                func_decls = tool.function_declarations
                assert len(func_decls) == 1
                func = func_decls[0]
                assert func.name == "get_weather"
                assert func.description == weather_tool.description
            return mock_response

        monkeypatch.setattr(
            provider.client.aio.models, "generate_content", mock_generate_content
        )

        # Test with tool
        response = await client.complete(
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            model="gemini-2.5-flash",
            tools=[weather_tool]
        )

        # Verify response contains tool calls
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments["location"] == "New York"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
