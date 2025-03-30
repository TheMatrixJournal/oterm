import inspect
import json
from ast import literal_eval  # 🧬 Safe parsing
from pathlib import Path
from typing import (Any, AsyncGenerator, AsyncIterator, Iterator, Literal, Mapping, Sequence)

from ollama import (AsyncClient, ChatResponse, Client, ListResponse, Message, Options, ProgressResponse, ShowResponse)
from pydantic.json_schema import JsonSchemaValue
from textual import log  # 🪲 Logging system

from oterm.config import envConfig  # 🧬 Centralized environment config
from oterm.types import ToolCall  # 🧬 Custom tool schema


def parse_format(format_text: str) -> JsonSchemaValue | Literal["", "json"]:
    try:
        jsn = json.loads(format_text)
        if isinstance(jsn, dict):  # ❓ Ensure valid dict format
            return jsn
    except json.JSONDecodeError:  # ☠️ Graceful fallback on bad format
        if format_text in ("", "json"):
            return format_text
    raise Exception(f"Invalid Ollama format: '{format_text}'")  # ☠️ Unhandled invalid format


class OllamaLLM:
    def __init__(  # 🧬 Main model configuration
        self,
        model="llama3.2",
        system: str | None = None,
        history: list[Mapping[str, Any] | Message] = [],
        format: str = "",
        options: Options = Options(),
        keep_alive: int = 5,
        tool_defs: Sequence[ToolCall] = [],
    ):
        self.model = model
        self.system = system
        self.history = history
        self.format = format
        self.keep_alive = keep_alive
        self.options = options
        self.tool_defs = tool_defs
        self.tools = [tool["tool"] for tool in tool_defs]  # 🧬 Tool integration

        if system:
            system_prompt: Message = Message(role="system", content=system)  # ❤️ Inject system prompt
            self.history = [system_prompt] + self.history


    async def completion(  # ⏳ Async chat interaction
        self,
        prompt: str = "",
        images: list[Path | bytes | str] = [],
        tool_call_messages=[],
    ) -> str:
        client = AsyncClient(  # ✨ Async Ollama client
            host=envConfig.OLLAMA_URL, verify=envConfig.OTERM_VERIFY_SSL
        )
        if prompt:
            user_prompt: Message = Message(role="user", content=prompt)
            if images:
                user_prompt.images = images  # ⚠️ Known Ollama image bug
            self.history.append(user_prompt)
        response: ChatResponse = await client.chat(  # ⏳ Await response
            model=self.model,
            messages=self.history + tool_call_messages,
            keep_alive=f"{self.keep_alive}m",
            options=self.options,
            format=parse_format(self.format),
            tools=self.tools,
        )
        message = response.message
        tool_calls = message.tool_calls
        if tool_calls:  # ❓ Check for tool use
            tool_messages = [message]
            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                for tool_def in self.tool_defs:
                    log.debug("Calling tool: %s", tool_name)  # 🪲 Tool debug log
                    if tool_def["tool"]["function"]["name"] == tool_name:
                        tool_callable = tool_def["callable"]
                        tool_arguments = tool_call["function"]["arguments"]
                        try:
                            if inspect.iscoroutinefunction(tool_callable):  # ❓ Async tool?
                                tool_response = await tool_callable(**tool_arguments)  # ⏳ Run tool
                            else:
                                tool_response = tool_callable(**tool_arguments)  # ⚠️ No type check
                            log.debug(f"Tool response: {tool_response}", tool_response)
                        except Exception as e:  # ☠️ Error handling
                            log.error(f"Error calling tool {tool_name}", e)
                            tool_response = str(e)
                        tool_messages.append(
                            {  # type: ignore
                                "role": "tool",
                                "content": tool_response,
                                "name": tool_name,
                            }
                        )
            return await self.completion(tool_call_messages=tool_messages)  # ♻️ Recursive response with tool replies

        self.history.append(message)
        return message.content or ""


    async def stream(  # ⏳ Streaming generator
        self,
        prompt: str,
        images: list[Path | bytes | str] = [],
        additional_options: Options = Options(),
        tool_defs: Sequence[ToolCall] = [],
    ) -> AsyncGenerator[str, Any]:
        if tool_defs:  # ❓ Not implemented yet
            raise NotImplementedError(
                "stream() should not be called with tools till Ollama supports streaming with tools."
            )

        client = AsyncClient(
            host=envConfig.OLLAMA_URL, verify=envConfig.OTERM_VERIFY_SSL
        )
        user_prompt: Message = Message(role="user", content=prompt)
        if images:
            user_prompt.images = images  # ⚠️ Known type-ignore
        self.history.append(user_prompt)

        options = {  # 🧬 Merge default and custom options
            k: v for k, v in self.options.model_dump().items() if v is not None
        } | {k: v for k, v in additional_options.model_dump().items() if v is not None}

        stream: AsyncIterator[ChatResponse] = await client.chat(
            model=self.model,
            messages=self.history,
            stream=True,
            options=options,
            keep_alive=f"{self.keep_alive}m",
            format=parse_format(self.format),
            tools=self.tools,
        )
        text = ""
        async for response in stream:  # ⏳ Real-time token output
            text = text + response.message.content if response.message.content else text
            yield text

        self.history.append(Message(role="assistant", content=text))


    @staticmethod
    def list() -> ListResponse:  # 🧬 List models
        client = Client(host=envConfig.OLLAMA_URL, verify=envConfig.OTERM_VERIFY_SSL)
        return client.list()


    @staticmethod
    def show(model: str) -> ShowResponse:  # 🧬 Show model details
        client = Client(host=envConfig.OLLAMA_URL, verify=envConfig.OTERM_VERIFY_SSL)
        return client.show(model)


    @staticmethod
    def pull(model: str) -> Iterator[ProgressResponse]:  # ⚡ Stream pull progress
        client = Client(host=envConfig.OLLAMA_URL, verify=envConfig.OTERM_VERIFY_SSL)
        stream: Iterator[ProgressResponse] = client.pull(model, stream=True)
        for response in stream:
            yield response


def parse_ollama_parameters(parameter_text: str) -> Options:  # 🧬 Parse user-defined params
    lines = parameter_text.split("\n")
    params = Options()
    valid_params = set(Options.model_fields.keys())
    for line in lines:
        if line:
            key, value = line.split(maxsplit=1)
            try:
                value = literal_eval(value)  # 🧬 Evaluate safe literal
            except (SyntaxError, ValueError):  # ☠️ Fallback on parse error
                pass
            if key not in valid_params:
                continue
            if params.get(key):  # ❓ Append if already exists
                if not isinstance(params[key], list):
                    params[key] = [params[key], value]
                else:
                    params[key].append(value)
            else:
                params[key] = value
    return params


def jsonify_options(options: Options) -> str:  # 🧬 JSON-ify options
    return json.dumps(
        {
            key: value
            for key, value in options.model_dump().items()
            if value is not None
        },
        indent=2,
    )