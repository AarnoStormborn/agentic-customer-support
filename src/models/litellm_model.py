
from typing import AsyncGenerator, List, Dict, Any, Optional
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types as genai_types
import litellm

class LiteLLMModel(BaseLlm):
    """
    Adapter for using LiteLLM supported models with Google ADK.
    """
    model: str
    
    def __init__(self, model: str):
         super().__init__(model=model)

    async def generate_content_async(
        self, 
        llm_request: LlmRequest, 
        stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        
        messages = self._convert_request_to_messages(llm_request)
        
        # Tools handling (basic placeholder for now)
        tools = self._convert_tools(llm_request.tools_dict) if llm_request.tools_dict else None
        
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                stream=stream
            )
            
            if stream:
                async for chunk in response:
                    yield self._convert_chunk_to_response(chunk)
            else:
                yield self._convert_response_to_response(response)
        except Exception as e:
            # Return error in response if possible
            yield LlmResponse(content=None, error_message=str(e))

    def _convert_request_to_messages(self, request: LlmRequest) -> List[Dict[str, Any]]:
        messages = []
        
        # Convert ADK request to LiteLLM messages
        if request.contents:
            for content in request.contents:
                role = content.role
                if role == "model":
                    role = "assistant"
                
                parts_content = ""
                if content.parts:
                    for part in content.parts:
                        if part.text:
                            parts_content += part.text
                        # Handle other parts if needed
                
                messages.append({
                    "role": role,
                    "content": parts_content
                })
        return messages

    
    def _convert_tools(self, tools_dict: Dict) -> Optional[List[Dict[str, Any]]]:
        if not tools_dict:
            return None
        
        openai_tools = []
        for name, tool in tools_dict.items():
            # Assume tool is a BaseTool subclass with _get_declaration()
            # The declaration is likely a google.genai.types.FunctionDeclaration (or Tool/Schema)
            
            # Based on inspection: declaration.name, declaration.description, declaration.parameters
            # parameters is a Schema object
            
            if hasattr(tool, '_get_declaration'):
                decl = tool._get_declaration()
                
                function_def = {
                    "name": decl.name,
                    "description": decl.description,
                }
                
                if decl.parameters:
                    # decl.parameters is Google Schema object. Need to convert to JSON Schema/generic dict
                    parameters = self._convert_schema(decl.parameters)
                    function_def["parameters"] = parameters
                
                openai_tools.append({
                    "type": "function",
                    "function": function_def
                })
        
        return openai_tools

    def _convert_schema(self, schema) -> Dict[str, Any]:
        """Convert Google GenAI Schema to JSON Schema dict."""
        json_schema = {}
        
        # Mapping types
        # Google Types: STRING, INTEGER, NUMBER, BOOLEAN, ARRAY, OBJECT
        type_str = str(schema.type).split('.')[-1] # Enum to string 'STRING'
        
        type_map = {
            'STRING': 'string',
            'INTEGER': 'integer',
            'NUMBER': 'number',
            'BOOLEAN': 'boolean',
            'ARRAY': 'array',
            'OBJECT': 'object'
        }
        
        # Try to lower case first if string match fails
        if type_str in type_map:
            json_schema['type'] = type_map[type_str]
        else:
            # Fallback or try direct string access
            lower_type = str(schema.type).lower()
            if 'integer' in lower_type: json_schema['type'] = 'integer'
            elif 'string' in lower_type: json_schema['type'] = 'string'
            elif 'number' in lower_type: json_schema['type'] = 'number'
            elif 'boolean' in lower_type: json_schema['type'] = 'boolean'
            elif 'array' in lower_type: json_schema['type'] = 'array'
            elif 'object' in lower_type: json_schema['type'] = 'object'

        if schema.description:
            json_schema['description'] = schema.description
            
        if schema.properties:
            props = {}
            for k, v in schema.properties.items():
                props[k] = self._convert_schema(v)
            json_schema['properties'] = props
            
        if schema.required:
            json_schema['required'] = list(schema.required)
            
        if schema.items:
             json_schema['items'] = self._convert_schema(schema.items)
             
        return json_schema

    def _convert_response_to_response(self, response) -> LlmResponse:
        choice = response.choices[0]
        message = choice.message
        content_text = message.content
        tool_calls = message.tool_calls
        finish_reason = choice.finish_reason # 'stop' or 'tool_calls'

        parts = []
        if content_text:
             parts.append(genai_types.Part(text=content_text))
             
        if tool_calls:
            for tc in tool_calls:
                # LiteLLM/OpenAI tool call: id, type='function', function={name, arguments(str)}
                # ADK expects FunctionCall part
                # FunctionCall(id=..., name=..., args={...})
                import json
                try:
                    args = json.loads(tc.function.arguments)
                except:
                    args = {}
                
                parts.append(genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id=tc.id,
                        name=tc.function.name,
                        args=args
                    )
                ))

        # Create Content object
        content = genai_types.Content(
            role="model",
            parts=parts
        )
        
        # If tool calls are present, we shouldn't necessarily mark turn_complete=True?
        # ADK might handle it.
        
        return LlmResponse(content=content) #, turn_complete=(finish_reason != 'tool_calls')) # Check if turn_complete is logical here

    def _convert_chunk_to_response(self, chunk) -> LlmResponse:
        content_text = chunk.choices[0].delta.content or ""
        
        content = genai_types.Content(
            role="model",
            parts=[genai_types.Part(text=content_text)]
        )
        
        return LlmResponse(content=content, partial=True)

