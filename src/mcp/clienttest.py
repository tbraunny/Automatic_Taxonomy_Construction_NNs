import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

import re
import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import timedelta
from ollama import chat
from ollama import ChatResponse

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Union

from utils.llm_service import load_environment_llm

from abc import ABC


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class ArgumentValues(ABC, BaseModel):
    argument: str
    value: str|int|float

class ToolResponse(ABC, BaseModel):
    tool: str #= ""
    arguments: List[ArgumentValues] #= Field(..., description="The arguments of the function") #= {}

class ChatResponse(ABC, BaseModel):
    response: str #= Field("",description="The response from the LLM")
    tool_call: Union[ToolResponse|None] #= Field(None,description="The tool call information if a tool is needed")
    need_to_call_tool: bool #= Field(description="Field used to determine if a tool call needs to be made")


class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = 'fff'

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the JSON configuration file.

        Returns:
            Dict containing server configuration.

        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            JSONDecodeError: If configuration file is invalid JSON.
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key.

        Returns:
            The API key as a string.

        Raises:
            ValueError: If the API key is not found in environment variables.
        """
        #if not self.api_key:
        #    raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.stdio_context: Any | None = None
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("The command must be a valid string and cannot be None.")

        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                # read_timeout_seconds with timedelta of 15 minutes
                ClientSession(read, write, read_timeout_seconds=timedelta(minutes=15))
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            await self.cleanup()
            raise

    async def list_resources(self) -> list[Any]:

        resources = await self.session.list_resources()
        #print(resources)
        #input()
        return list(resources)

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()

        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                for tool in item[1]:
                    tools.append(Tool(tool.name, tool.description, tool.inputSchema))

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                #test = await self.session.call_tool("sparql_query", arguments=arguments)
                result = await self.session.call_tool(tool_name, arguments)
                return result

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.stdio_context = None
            except Exception as e:
                logging.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key
        self.llm = load_environment_llm().llm.with_structured_output(ChatResponse)

    def get_response(self, messages: list[dict[str, str]]) -> ChatResponse:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """
        response = self.llm.invoke(messages)
        return response


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client

        self.messages = []
        

    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = []
        for server in self.servers:
            cleanup_tasks.append(asyncio.create_task(server.cleanup()))

        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logging.warning(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: ChatResponse) -> str|ChatResponse:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            if llm_response.need_to_call_tool:
                logging.info("Processing LLM response for tool execution...")
                tool_call = llm_response.tool_call

                #tool_call = tool_call.dict() if isinstance(tool_call, ToolResponse) else tool_call
                if tool_call:
                    tool_call_input = {}
                    tool_call_input["tool"] = tool_call.tool
                    tool_call_input["arguments"] = {}

                    for i in tool_call.arguments:
                        tool_call_input["arguments"][i.argument] = i.value
                    
                    tool_call = tool_call_input

                #tool_call = json.loads(llm_response)
                if "tool" in tool_call and "arguments" in tool_call:
                    logging.info(f"Executing tool: {tool_call['tool']}")
                    logging.info(f"With arguments: {tool_call['arguments']}")

                    for server in self.servers:
                        tools = await server.list_tools()
                        if any(tool.name == tool_call["tool"] for tool in tools):
                            try:
                                result = await server.execute_tool(
                                    tool_call["tool"], tool_call["arguments"]
                                )

                                if isinstance(result, dict) and "progress" in result:
                                    progress = result["progress"]
                                    total = result["total"]
                                    percentage = (progress / total) * 100
                                    logging.info(
                                        f"Progress: {progress}/{total} "
                                        f"({percentage:.1f}%)"
                                    )
                                return f"Tool execution result: {result}"
                            except Exception as e:
                                error_msg = f"Error executing tool: {str(e)}"
                                logging.error(error_msg)
                                return error_msg

                    return f"No server found with tool: {tool_call['tool']}"
                return llm_response
        except json.JSONDecodeError:
            return llm_response
        return llm_response
    async def initialize(self):
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            self.all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                self.all_tools.extend(tools)
        except:
            pass

    async def start(self) -> None:
        """Main chat session handler."""
        try:
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"Failed to initialize server: {e}")
                    await self.cleanup_servers()
                    return

            tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools])
            self.system_message = (
                    "You are a helpful assistant with access to these tools:\n\n"
                    f"{tools_description}\n"
                    "Choose the appropriate tool based on the user's question. "
                    f"Reply with the following format: {{ChatResponse.schema_json(indent=2)}}\n\n"
                    "If no tool is needed, reply directly and fill in the response field.\n\n"
                    "After receiving a tool's response:\n"
                    "1. Transform the raw data into a natural, conversational response\n"
                    "2. Keep responses concise but informative\n"
                    "3. Focus on the most relevant information\n"
                    "4. Use appropriate context from the user's question\n"
                    "5. Avoid simply repeating the raw data\n\n"
                    "6. Reply in the response field with your output.\n"
                    "Please use only the tools that are explicitly defined above."
                )
            


            self.messages = [{"role": "system", "content": self.system_message}]

            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        logging.info("\nExiting...")
                        break

                    self.messages.append({"role": "user", "content": user_input})
                    

                    llm_response = self.llm_client.get_response(self.messages)
                    logging.info("\nAssistant: %s", llm_response.response)\
    
                    result = await self.process_llm_response(llm_response)
                    

                    if result != llm_response:
                        self.messages.append({"role": "assistant", "content": str(llm_response.tool_call)})
                        self.messages.append({"role": "user", "content": result})

                        final_response = self.llm_client.get_response(self.messages)
                        logging.info("\nFinal response: %s", str(final_response.response))
                        self.messages.append(
                            {"role": "assistant", "content": str(final_response.response)}
                        )
                    else:
                        self.messages.append({"role": "assistant", "content": str(llm_response)})

                except KeyboardInterrupt:
                    logging.info("\nExiting...")
                    break

        finally:
            await self.cleanup_servers()
    async def query_server(self, message):
        tools_description = "\n".join([tool.format_for_llm() for tool in self.all_tools])
        self.system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                f"Reply with the following format: {{ChatResponse.schema_json(indent=2)}}\n\n"
                "If no tool is needed, reply directly and fill in the response field.\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "6. Reply in the response field with your output.\n"
                "Please use only the tools that are explicitly defined above."
            )
        
        if len(self.messages) == 0:
            self.messages.append({"role": "system", "content": self.system_message})
        output = []
        try:
            self.messages.append({"role": "user", "content": message})
            llm_response = self.llm_client.get_response(self.messages)
            #logging.info("\nAssistant: %s", llm_response.response)
            
            result = await self.process_llm_response(llm_response)
            output.append(llm_response.response)

            if result != llm_response:
                self.messages.append({"role": "assistant", "content": str(llm_response.tool_call)})
                self.messages.append({"role": "user", "content": result})

                final_response = self.llm_client.get_response(self.messages)
                #logging.info("\nFinal response: %s", str(final_response.response))
                output.append(final_response.response)
                self.messages.append(
                    {"role": "assistant", "content": str(final_response.response)}
                )
            else:
                self.messages.append({"role": "assistant", "content": str(llm_response)})
        except:
            print(f"EXCEPTION: {e}")
        return output


async def main() -> None:
    """Initialize and run the chat session."""
    config = Configuration()
    server_config = config.load_config(f"{os.path.dirname(os.path.abspath(__file__))}/servers_config.json")
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()
    ]
    llm_client = LLMClient(config.llm_api_key)
    chat_session = ChatSession(servers, llm_client)
    await chat_session.initialize()
    #await chat_session.start()
    while True:
        query = input("User:")
        output = await chat_session.query_server(query)
        print("LLM: ", output)


if __name__ == "__main__":
    asyncio.run(main())
