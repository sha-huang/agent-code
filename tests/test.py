from agent_code.agent import run_agent
from agent_code.model import MockProvider
from agent_code.tools import default_tools


def test_echo_loop() -> None:
    result = run_agent("echo this message", MockProvider(), default_tools())

    assert "tool_call: echo" in result.trace[0]
    assert "observation: echo this message" in result.trace[1]
    assert result.final == "echo this message"


def test_mockprovider_messages() -> None:
    result = run_agent("echo this message", MockProvider(), default_tools())

    assert result.messages[1]["role"] == "model"
    assert "function_call" in result.messages[1]["parts"][0]
    assert result.messages[1]["parts"][0]["function_call"]["name"] == "echo"
    assert result.messages[1]["parts"][0]["function_call"]["id"] == "call_echo_1"

    assert result.messages[2]["role"] == "user"
    assert "function_response" in result.messages[2]["parts"][0]
    assert result.messages[2]["parts"][0]["function_response"]["name"] == "echo"
    assert result.messages[2]["parts"][0]["function_response"]["id"] == "call_echo_1"