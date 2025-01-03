from typing import Sequence

from oterm.tools.date_time import DateTimeTool, date_time
from oterm.tools.location import LocationTool, current_location
from oterm.tools.shell import ShellTool, shell_command
from oterm.tools.weather import WeatherTool, current_weather
from oterm.tools.web import WebTool, fetch_url
from oterm.types import ToolDefinition

available: Sequence[ToolDefinition] = [
    {"tool": DateTimeTool, "callable": date_time},
    {"tool": ShellTool, "callable": shell_command},
    {"tool": LocationTool, "callable": current_location},
    {"tool": WeatherTool, "callable": current_weather},
    {"tool": WebTool, "callable": fetch_url},
]
