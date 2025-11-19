"""
LLM Client for interacting with Claude or GPT models
"""
import json
from typing import List, Dict, Any, Optional
from anthropic import Anthropic
from openai import OpenAI
from config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
)


class LLMClient:
    """Client for interacting with LLM APIs"""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or LLM_PROVIDER

        if self.provider == "anthropic":
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = ANTHROPIC_MODEL
        elif self.provider == "openai":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Send a chat request to the LLM

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            max_tokens: Maximum tokens in response

        Returns:
            Response dict with content and tool calls
        """
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, max_tokens)
        elif self.provider == "openai":
            return self._chat_openai(messages, tools, max_tokens)

    def _chat_anthropic(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Chat with Claude API"""
        # Anthropic expects system message separately
        system_message = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": filtered_messages,
        }

        if system_message:
            kwargs["system"] = system_message

        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)

        # Parse response
        result = {
            "content": "",
            "tool_calls": [],
            "stop_reason": response.stop_reason,
        }

        for block in response.content:
            if block.type == "text":
                result["content"] += block.text
            elif block.type == "tool_use":
                result["tool_calls"].append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return result

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Chat with OpenAI API"""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        if tools:
            # Convert to OpenAI tool format
            openai_tools = [
                {"type": "function", "function": tool} for tool in tools
            ]
            kwargs["tools"] = openai_tools

        response = self.client.chat.completions.create(**kwargs)

        message = response.choices[0].message

        result = {
            "content": message.content or "",
            "tool_calls": [],
            "stop_reason": response.choices[0].finish_reason,
        }

        if message.tool_calls:
            for tool_call in message.tool_calls:
                result["tool_calls"].append({
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                })

        return result


def create_tool_definition(
    name: str,
    description: str,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Create a tool definition for LLM function calling

    Args:
        name: Tool name
        description: What the tool does
        parameters: JSON schema for parameters

    Returns:
        Tool definition dict
    """
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": parameters,
            "required": list(parameters.keys()),
        },
    }
