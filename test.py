from litellm import completion
from dataclasses import dataclass
import inspect
from pprint import pprint
import json

@dataclass
class Event:
    pass

@dataclass
class PartialEvent(Event):
    pass

@dataclass
class FullEvent(Event):
    pass

@dataclass
class TextDelta(PartialEvent):
    content: str

@dataclass
class ReasoningDelta(PartialEvent):
    content: str

@dataclass
class ToolCallSelect(PartialEvent):
    function_name: str

@dataclass
class ToolCallArguments(PartialEvent):
    arguments_content: str

@dataclass
class TextResponse(FullEvent):
    content: str

@dataclass
class ReasoningResponse(FullEvent):
    content: str

@dataclass
class ToolCallRequest(FullEvent):
    function_name: str
    arguments: dict

@dataclass
class End(FullEvent):
    finish_reason: str


def get_json_schema_name(arg_type: str) -> str:
    """Convert a Python type to a JSON schema type."""
    match arg_type:
        case "int":
            return "integer"
        case "float":
            return "number"
        case "str":
            return "string"
        case "bool":
            return "boolean"
        case "dict":
            return "object"
        case "list":
            return "array"
        case "tuple":
            return "array"
        case "set":
            return "array"
        case _:
            return "object"  


def describe_function(func, description: str = None):
    function_name = func.__name__
    argument_specs = {key: value.__name__ for key, value in inspect.getfullargspec(func).annotations.items()}

    if "return" in argument_specs.keys():
        argument_types = {key: get_json_schema_name(value) for key, value in argument_specs.items() if key != "return"}
        return_type = get_json_schema_name(argument_specs["return"])
    else:
        argument_types = {key: get_json_schema_name(value) for key, value in argument_specs.items()}
        return_type = None

    properties = {key: {"type": value} for key, value in argument_types.items() if value}

    function_description = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": description or "",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties.keys()),
                },
            },
        }
    return function_description


def process_response(response):

    full_text_content = ""
    full_reasoning_content = ""

    tool_name = None
    tool_arguments = ""

    current_event = None
    prev_event = None

    for chunk in response:

        choices = chunk.choices

        for choice in choices:
            delta = choice.delta

            content = delta.content
            if content:
                current_event = TextDelta(content=content)

            try:
                reasoning_content = delta.reasoning_content
            except AttributeError:
                reasoning_content = None
            if reasoning_content:
                current_event = ReasoningDelta(content=reasoning_content)

            tool_calls = delta.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    arguments_content = tool_call.function.arguments
                    function_name = tool_call.function.name

                    if function_name:
                        current_event = ToolCallSelect(function_name=function_name)

                    if arguments_content:
                        current_event = ToolCallArguments(arguments_content=arguments_content)

            finish_reason = choice.finish_reason
            if finish_reason:
                current_event = End(finish_reason=finish_reason)

            full_event = None
            if isinstance(current_event, TextDelta) and not prev_event:
                full_text_content = current_event.content 

            if isinstance(current_event, ReasoningDelta) and not prev_event:
                full_reasoning_content = current_event.content

            if isinstance(current_event, TextDelta) and isinstance(prev_event, TextDelta):
                full_text_content += current_event.content

            if isinstance(current_event, ReasoningDelta) and isinstance(prev_event, ReasoningDelta):
                full_reasoning_content += current_event.content

            if isinstance(prev_event, TextDelta) and not isinstance(current_event, TextDelta):
                full_event = TextResponse(content=full_text_content)
                full_text_content = ""

            if isinstance(prev_event, ReasoningDelta) and not isinstance(current_event, ReasoningDelta):
                full_event = ReasoningResponse(content=full_reasoning_content)
                full_reasoning_content = ""

            if isinstance(current_event, ToolCallSelect):
                tool_name = current_event.function_name

            if isinstance(current_event, ToolCallArguments):
                tool_arguments += current_event.arguments_content

            if isinstance(prev_event, ToolCallArguments) and not isinstance(current_event, ToolCallArguments):
                full_event = ToolCallRequest(function_name=tool_name, arguments=tool_arguments)
                tool_arguments = ""
                tool_name = None

        prev_event = current_event

        if not isinstance(current_event, End):
            yield current_event

        if full_event:
            yield full_event

        # if isinstance(current_event, End):
        #     yield current_event



@dataclass
class LLMModelEndpoint:
    model: str
    api_key: str
    api_base: str


class Agent:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.tools = {}
        self.tool_descriptions = {}

    def add_function_tool(self, function, function_description: dict):
        tool_description = describe_function(function, function_description)
        tool_name = tool_description["function"]["name"]
        self.tool_descriptions[tool_name] = tool_description
        self.tools[tool_name] = function

    def make_request(self, user_prompt: str, model_endpoint: LLMModelEndpoint, temperature: float | None = None, top_p: float | None = None):
        response = completion(
            model=model_endpoint.model,
            api_key=model_endpoint.api_key,
            api_base=model_endpoint.api_base,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=self.tool_descriptions.values() if self.tool_descriptions else None,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        )
        for event in process_response(response):
            yield event

    def execute_tool(self, tool_name: str, arguments_json_string: str):
        if tool_name not in self.tools:
            raise ValueError(f"Agent called tool: {tool_name} which is not available.")

        try:
            arguments = json.loads(arguments_json_string)
        except json.JSONDecodeError as e:
            raise ValueError(f"Agent created invalid JSON with args for tool: {tool_name}: {arguments_json_string}") from e
        
        try:
            function_result = self.tools[tool_name](**arguments)
        except Exception as e:
            raise ValueError(f"Agent failed to execute tool: {tool_name} with args: {arguments}") from e

        return function_result

""" END OF AGENT FRAMEWORK CODE """


def get_city_temperature(city_name: str) -> str:
    return f"The current temperature in {city_name} is 20 degrees Celsius."

def get_city_rainfall(city_name: str) -> str:
    return f"The current rainfall in {city_name} is 5.0 mm."


my_agent = Agent(
    system_prompt=
        """
        You are a helpful assistant. You can answer questions and use tools to get information. 
        DO NOT make up information, only use the tools provided to you. 
        DO NOT USE TOOLS IF THEY ARE NOT HELPFUL.
        """
    )
my_agent.add_function_tool(get_city_temperature, "Function used to get the current temperature in a given city")
my_agent.add_function_tool(get_city_rainfall, "Function used to get the current rainfall in a given city")

for event in my_agent.make_request(
    user_prompt="Check the temperature in Paris.",
    # user_prompt="What is RAM?",
    model_endpoint=LLMModelEndpoint(
        model="openai/Qwen2-VL-2B-Instruct-Q6_K.gguf",
        api_key="sk-1234",
        api_base="http://0.0.0.0:8080",
    ),
    temperature=0.0,
    top_p=0.9,
):
    
    if isinstance(event, PartialEvent):
        print(f"Partial event: {event}")

    if isinstance(event, FullEvent):
        print(f"Full event: {event}")

    if isinstance(event, ToolCallRequest):
        result = my_agent.execute_tool(event.function_name, event.arguments)
        print(f"Tool {event.function_name} executed with result: {result}")
