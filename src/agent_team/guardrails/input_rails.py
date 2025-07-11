
from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    GuardrailFunctionOutput,
    input_guardrail,
    RunContextWrapper,
    TResponseInputItem
)

#### Supervisor Agent ####

class InputAttack(BaseModel):
    reasoning: str
    is_input_attack: bool
    

def init_supervisor_guardrail(name: str, prompt: str):
    
    guardrail_agent = Agent(
        name=name,
        instructions=prompt,
        output_type=InputAttack
    )
    
    @input_guardrail
    async def supervisor_guardrail(
        ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
    ):
        result = await Runner.run(guardrail_agent, input, context=ctx.context)
        
        return GuardrailFunctionOutput(
            output_info=result.final_output,
            tripwire_triggered=result.final_output.is_input_attack
        )
        
    return supervisor_guardrail
    

