# test context passing options

from dataclasses import dataclass
from typing import Optional, Type

from pydantic import BaseModel, PositiveFloat, Field, create_model
from pydantic_ai import Agent, RunContext
import logfire

import dotenv


dotenv.load_dotenv()

# a dependency model to be passed as context
# we could put dynamic default values here to be shared across agents
@dataclass
class AgentDeps:
    default_color = "fuschia"
    default_address = "456 Elm St"

# baseline -> static model definition
class UserInfo(BaseModel):
    # define output model here, focus on description metadata
    # default values will be provided dynamically
    name: str = Field( description="The user's full name.")
    phone: Optional[str] = Field( description="The user's phone number, if known.")
    address: Optional[str] = Field( description="The user's address, if known.")
    favorite_color: Optional[str] = Field( description="The user's favorite color, if known.")

# alternate 1 -> dynamic model definition with default values
UserInfoDynamic = create_model(
    'UserInfoDynamic',
    name=(str, "John Doe"),
    phone=(Optional[str], "888-6767"),
    address=(Optional[str], None),
    favorite_color=(Optional[str], None),
)

# alternate 2 -> factory function to create dynamic model definition
# function to create output model BaseModel CLASS with runtime different default values
# makes it easy to add description metadata
def make_user_info_model(default_phone: str, default_favorite_color: Optional[str] = "azure") -> type[BaseModel]:
    return create_model(
        "UserInfo",
        name=(str, Field(default="Jane Doe", description="The user's full name.")),
        phone=(Optional[str], Field(default=default_phone, description="The user's phone number.")),
        address=(Optional[str], Field(default=None, description="The user's address.")),
        favorite_color=(str, Field(default=default_favorite_color, description="The user's favorite color.")),
    )

UserInfoDynamic2 = make_user_info_model("777-1234", "magenta")
UserInfoDynamic3 = make_user_info_model("444-5678")
print("UserInfoDynamic3 model:", UserInfoDynamic3) # returns  <class '__main__.UserInfo'>

# alternate 3 -> factory function to return a model INSTANCE with runtime different default values
# this factory function creates an INSTANCE of UserInfo using context data
# allows the RunContext to provide the default values at runtime
# may be the most flexible approach, since the run context can be used in other places
def make_user_info_instance_using_context(context: RunContext[AgentDeps]) -> BaseModel:
    # Access dynamic data from the context and create the model instance
    
    # create an instance with desired default values, preserves descriptions too
    ui = UserInfo(name="Jane Doe",
                    phone="55555-12121212",
                    address=context.deps.default_address,
                    favorite_color=context.deps.default_color)
    print("Created UserInfo instance:", ui) # returns UserInfo instance with new values
    return ui

# NOPE: UserInfo.model_fields['phone'].default = '999-9999'  # alternative way to set default

class UserLocationContext(BaseModel):
    city: str = Field(default="Stilwell", description="The city where the user lives.")
    state: str = Field(default="KS", description="The state where the user lives.")
    country: str = Field(default="USA", description="The country where the user lives.")

model_string =  "openai:gpt-5.1" # "openai-responses:gpt-5.1" #'groq:llama-3.3-70b-versatile' # "gemini-3-pro-preview" #"openai:gpt-5.1"

logfire.configure()
logfire.instrument_pydantic_ai()

user_location_context = UserLocationContext()

agent = Agent(
    model=model_string,
    instructions="answer the user's questions based on their user info",
    #output_type=UserInfoDynamic3, # this works 
    output_type=make_user_info_instance_using_context, # this works, and leverages RunContext
    deps_type=AgentDeps,
    )

deps = AgentDeps() # set up dependencies with default values

response = agent.run_sync([
            "what is my name and phone number, and where do I live?.",
            # "user's current info: " + user_location_context.model_dump_json()  # this works fine
            ],
            deps=deps # the actual RunContext does NOT appear to get passed to the AI
            )

print("Agent response:", response.output)