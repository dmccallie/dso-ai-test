# allow for forward references
from __future__ import annotations
from dataclasses import dataclass
import datetime
from zoneinfo import ZoneInfo

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel
import logfire
from pydantic import BaseModel, Field
from typing import Union
from pathlib import Path

from ai_astronomy_utils import ai_localize_and_fetch_dsos
from ai_data_models import AstroDependencies, SA_Plan
from ai_data_models import model_string, model_settings

from agents import convert_local_day_and_time_to_utc_iso, infer_equipment_specs, \
    infer_observer_context, return_dsos_observer_gear , single_agent_astro_plan
    
# fetch openai api key from env file
import dotenv
dotenv.load_dotenv()

# logire notes:
# Your Logfire credentials are stored in /home/david/.logfire/default.toml
# setup using logfire-cli via uv
# uv add logfire
# uv run logfire auth
# uv run logfire projects use astro-planner-project
#  https://logfire-us.pydantic.dev/dmccallie/astro-planner-project

logfire.configure()
logfire.instrument_pydantic_ai()
# logfire.instrument_httpx(capture_all=True)



async def main() -> None:
    """Run a tiny REPL so we can test astro-agent."""

    import asyncio

    print("=== astro-agent demo ===")
    print("Type 'exit' or Ctrl+C to quit.\n")

    user_requests = []
    while True:
        try:
            if len(user_requests) == 0:
                new_msg = input("Please enter anew query: ")
                if new_msg.strip() == "":
                    continue
                if new_msg.lower() in {"exit", "quit"}:
                    print("Bye!")
                    break
            else:
                # print("\nPrevious conversation:" )   
                # for i, msg in enumerate(conversation):
                #     print(f"{i+1}: {msg}")
                new_msg = input("Refine your query or return to start new query?: ")
                if new_msg.strip() == "":
                    user_requests = []  # reset conversation on empty input
                    print("Conversation reset.\n")
                    continue

            user_requests.append(new_msg)

            # build query string. First message is the initial query, subsequent messages are updates to that query.
            if len(user_requests) == 1:
                query = user_requests[0]
            else:
                query = user_requests[0] + "\n"
                for update in user_requests[1:]:
                    query += "Update: " + update + "\n"
                query += "Otherwise no updates to the initial query."
        
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        try:
            if len(user_requests) > 0:
                print("\n--- Continuing conversation with context ---\n")
                print("Previous conversation:" )   
                for i, msg in enumerate(user_requests):
                    print(f"{i+1}: {msg[0:200]}")
            else:
                print("\n--- New conversation ---\n")
            
            # the dependencies are only available via RunContext in tools!
            # tools must be registered with takes_ctx=True to access deps
            updated_deps = AstroDependencies(
                # make sure these defaults are 'now' at runtime
                # note this should be CLIENT "now" not server!
                    default_time=datetime.datetime.now(ZoneInfo("America/Chicago")).strftime("%H:%M"),
                    default_date=datetime.datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d"),
                    default_timezone="America/Chicago",
                    default_location="Powell Observatory, Kansas", # this should be findable 
                    # default_latitude=38.7076,
                    # default_longitude=-94.7073,
                    default_telescope="Astrophysics 130EDF F6.3",
                    default_camera="ZWO ASI 2600MC Pro",
            )
            result = await single_agent_astro_plan.run(query, deps=updated_deps)
                            
            print("FINAL Bot Output --->:")
            if isinstance(result.output, SA_Plan):

                print(f"Generated query: {result.output.sql_query}")
                print(f"Equipment included in plan: {result.output.equipment}")
                print(f"Observer Context included in plan: {result.output.observer_context}")
                print(f"Is this a valid plan? {result.output.valid_plan}")

                # test running the generated SQL query against local DB
                db_path = Path(updated_deps.db_path)
                latidude = result.output.observer_context.latitude_deg
                longitude = result.output.observer_context.longitude_deg
                if latidude is None or longitude is None:
                    print("Cannot run local DSO query because latitude or longitude is missing in observer context.")
                    continue
                observe_date = result.output.observer_context.observe_date
                observe_time = result.output.observer_context.observe_time
                timezone = result.output.observer_context.timezone
                if observe_date is None or observe_time is None or timezone is None:
                    print("Cannot run local DSO query because observe date, time, or timezone is missing in observer context.")
                    continue

                dsl_results = ai_localize_and_fetch_dsos(
                    result.output.sql_query,
                    db_path,
                    latidude,
                    longitude,
                    observe_date,
                    observe_time,
                    timezone,
                )
                print(f"DSO Query Results ({len(dsl_results)} objects):")
                for i, dso in enumerate(dsl_results):
                    print(f"{i+1} - {dso['name']} ({dso['catalog']}) ({dso['class']} {dso['type']}), Altitude: {dso['altitude']:.1f} deg, Azimuth: {dso['azimuth']:.1f} deg")
                    # pairs = [item for item in dso.items()]
                    # print("  " + ", ".join([f"{k}: {v}" for k, v in pairs]))                    
            
            else:
                print(f"Output was NOT a PLAN: {result.output}")

        except Exception as e:
            print(f"Agent throws Error: {e}\n")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
