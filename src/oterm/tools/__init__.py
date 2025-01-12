from importlib import import_module
from typing import Awaitable, Callable, Sequence

from mcp import StdioServerParameters
from ollama._types import Tool

from oterm.config import appConfig
from oterm.tools.date_time import DateTimeTool, date_time
from oterm.tools.location import LocationTool, current_location
from oterm.tools.mcp import MCPClient
from oterm.tools.shell import ShellTool, shell_command
from oterm.tools.weather import WeatherTool, current_weather
from oterm.tools.web import WebTool, fetch_url
from oterm.types import ExternalToolDefinition, ToolDefinition


def load_tools(tool_defs: Sequence[ExternalToolDefinition]) -> Sequence[ToolDefinition]:
    tools = []
    for tool_def in tool_defs:
        tool_path = tool_def["tool"]

        try:
            module, tool = tool_path.split(":")
            module = import_module(module)
            tool = getattr(module, tool)
            if not isinstance(tool, Tool):
                raise Exception(f"Expected Tool, got {type(tool)}")
        except ModuleNotFoundError as e:
            raise Exception(f"Error loading tool {tool_path}: {str(e)}")

        callable_path = tool_def["callable"]
        try:
            module, function = callable_path.split(":")
            module = import_module(module)
            callable = getattr(module, function)
            if not isinstance(callable, (Callable, Awaitable)):
                raise Exception(f"Expected Callable, got {type(callable)}")
        except ModuleNotFoundError as e:
            raise Exception(f"Error loading callable {callable_path}: {str(e)}")
        tools.append({"tool": tool, "callable": callable})

    return tools


available: Sequence[ToolDefinition] = [
    {"tool": DateTimeTool, "callable": date_time},
    {"tool": ShellTool, "callable": shell_command},
    {"tool": LocationTool, "callable": current_location},
    {"tool": WeatherTool, "callable": current_weather},
    {"tool": WebTool, "callable": fetch_url},
]

external_tools = appConfig.get("tools")
if external_tools:
    available.extend(load_tools(external_tools))


async def setup_mcp_servers():
    mcp_servers = appConfig.get("mcpServers")
    if mcp_servers:

        for server, config in mcp_servers.items():
            async with MCPClient(
                StdioServerParameters.model_validate(config)
            ) as client:
                tools = await client.get_available_tools()
                print(tools)
