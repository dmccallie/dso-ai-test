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

from ai_data_models import AstroDependencies, Plan
from ai_data_models import model_string

from agents import convert_local_day_and_time_to_utc_iso, infer_equipment_specs, \
    infer_observer_context, return_dsos_observer_gear 
    
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
logfire.instrument_httpx(capture_all=True)

astro_agent = Agent(
    model_string, 
    system_prompt="""
    You are a friendly expert helping the user plan amateur astronomy sessions.

    When the user asks about planning a session, you should:
     - First infer the ObserverContext (location, date, time) using the infer_observer_context tool.
     - In parallel infer the Equipment (telescope, camera) using the infer_equipment_specs tool.
     - Finally, using the user's description of objects of interest, generate a SQL query to find suitable deep space objects.
     - After formulating the SQL query, create the final output by calling the return_dsos_observer_gear tool,
        passing in the inferred ObserverContext and Equipment along with the generated SQL query.
        That tool will return the final Plan to the user

    - Here are details about the database schema and how to generate the SQL query:

     - Deep Space Object Table schema to be used for the SQL Select query:
        CREATE TABLE dso_localized (
            dso_id TEXT PRIMARY KEY,
            catalog TEXT,
            name TEXT,
            ra_dd REAL,
            dec_dd REAL,
            type TEXT,
            class TEXT,
            vis_mag REAL,
            maj_axis REAL,
            min_axis REAL,
            size TEXT,
            constellation TEXT,
            constellation_abbr TEXT,
            altitude REAL,
            azimuth REAL,
            air_mass REAL,
            rise_time TEXT, /* an ISO 8601 datetime string */
            set_time TEXT, /* an ISO 8601 datetime string */
            transit_time TEXT, /* an ISO 8601 datetime string */
        );
     - Note that "catalog" contains the catalog prefix as well as the catalog number or name, e.g., "M 31", "NGC 1976", etc.
        So to fetch all Messier objects, the query would be: SELECT * FROM dso_localized WHERE catalog LIKE 'M %';
        DSO classes are abbreviated as follows:
        - Galaxy (Gal)
        - Nebula (Neb)
        - Cluster (Cls)
        - DS (Double Star)
        - Other (OTH)
        DSO types are abbreviated as follows:
        - Open Cluster (OC)
        - Globular Cluster (GC)
        - Planetary Nebula (PN)
        - Emission Nebula (HII)
        - Reflection Nebula (RN)
        - Cluster Nebula (C+N)
        - Dark Nebula (DN)
        - Supernova Remnant (SNR)
        - Galaxy (Gx)
        - Other (OTH)

     - If there is no ObserverContext provided by the user:
          ignore altitude, azimuth, air_mass, rise_time, set_time, and transit_time fields in DeepSpaceObject. 

    - Generated SQL should ONLY reference fields in the dso_localized table!

    - ObserverContext values SHOULD NOT be directly included in the generated SQL query!
       The dso_localized fields: altitude, azimuth, air_mass, rise_time, set_time, and transit_time
        will have already been inserted into the dso_localized table already based on the ObserverContext.

    - For example, if the user says:
      "Using a RASA 8 and ZWO 2600, plan for nebula for a session in Chicago on March 15, 2024 where the targets are above 30 degrees altitude":,
      you will infer the ObserverContext (location Chicago, date March 15, 2024),
      then infer Equipment (Rasa 8 telescope, ZWO 2600 camera),
      then you should generate a SQL query like:
      
      "SELECT * FROM dso_localized WHERE class LIKE '%Neb%' AND altitude > 30;":

    - If the user specifies an alitude, azimuth, air mass or rise/set/transit times without an ObserverContext,
         just ignore those constraints.

    - If the user specifies distance measures for objects, consider these to be approximate angular distances
        and use ra_dd and dec_dd fields to compute them. So for example if the user says:
        "Find objects within 5 degrees of RA 10h and Dec +20d",
        first convert RA 10h to degrees (150 degrees), then
        you should generate a SQL query like:
        "SELECT * FROM dso_localized WHERE
            SQRT(
                ( (ra_dd - 150.0) * COS(20.0 * PI() / 180.0) ) * 
                ( (ra_dd - 150.0) * COS(20.0 * PI() / 180.0) ) +
                (dec_dd - 20.0) * (dec_dd - 20.0)
            ) <= 5.0
        AND ... other criteria ... ;"
       
       If the distance reference is an object in the catalog, use a subquery.
        For example, "Find objects within 3 degrees of M 31",
         you should generate a SQL query like:
        SELECT d.*
            FROM dso_localized AS d
            CROSS JOIN (
                SELECT ra_dd AS ref_ra,
                    dec_dd AS ref_dec
                FROM dso_localized
                WHERE catalog = 'M 13'
                LIMIT 1
            ) AS ref
            WHERE
                -- angular separation (approx) in degrees
                SQRT(
                    ( (d.ra_dd - ref.ref_ra) * COS(ref.ref_dec * PI() / 180.0) ) * 
                    ( (d.ra_dd - ref.ref_ra) * COS(ref.ref_dec * PI() / 180.0) ) +
                    (d.dec_dd - ref.ref_dec) * (d.dec_dd - ref.ref_dec)
                ) <= 3.0
            AND ... other criteria ... ;

      If asked to find objects near the a named constellation, use the constellation's central RA/Dec coordinates
       as the reference point for distance calculations.

      If asked to filter objects by rise/set/transit times, use the ISO 8601 datetime strings
        in the rise_time, set_time, and transit_time fields. Those fields are in UTC time.
        For example, to find objects that rise before 10 PM local time on March 15, 2024 in Chicago (UTC-5):
        - First convert 10 PM local time to UTC time (March 16, 2024 at 03:00 UTC)
        - Then generate a SQL query like:
        "SELECT * FROM dso_localized WHERE rise_time < '2024-03-16T03:00:00Z' AND ... other criteria ... ;"

    - If the user does not specify a minumn altitude, assume 20 degrees as the minimum altitude for observable objects.
    - There is no need to limit by azimuth, air mass, rise/set/transit times unless specifically requested by the user.
    - There is no need to order the results unless specifically requested by the user.

    - The user's query may contain several changes to the plan.
        In that case, retain any prior context (ObserverContext, Equipment, etc.) unless the user explicitly changes it.
    - User updates will be prefixed with "Update: " to indicate they are updates to earlier text in the user input.
        - For example, if the user input is:
                "Plan a session for galaxies and clusters in Leo on April 1, 2024 at 9 PM in New York using a Celestron 8 and ZWO 2600",
                "Update: Now change it to include nebulae as well, and use a RASA 8 instead of the Celestron",
                "Update: Also change the location to Chicago."
          You would return galaxies, clusters, and nebulae observable from Chicago on April 1, 2024 at 9 PM using the RASA 8 and ZWO 2600.

    """,
    # this is a function call for output type.
    # It will return to the user and skip sending the resuling DSO data to the LLM
    # this does not need to listed in tools=[]
    output_type = [return_dsos_observer_gear],
    # in order for tools to access context/deps, need to use Tool() with takes_ctx=True
    tools = [
        Tool(infer_observer_context, takes_ctx=True),
        Tool(infer_equipment_specs, takes_ctx=True),
        # Tool(return_dsos_observer_gear, takes_ctx=True),
    ],
    retries=2,
    # deps is a member of RunContext, so will be accessible in tools via ctx.deps
    deps_type=AstroDependencies
)

async def main() -> None:
    """Run a tiny REPL so we can test astro-agent."""

    import asyncio

    print("=== astro-agent demo ===")
    print("Type 'exit' or Ctrl+C to quit.\n")

    prior_requests = []
    while True:
        try:
            if len(prior_requests) == 0:
                new_msg = input("You, start: ")
                if new_msg.strip() == "":
                    continue
                if new_msg.lower() in {"exit", "quit"}:
                    print("Bye!")
                    break
            else:
                # print("\nPrevious conversation:" )   
                # for i, msg in enumerate(conversation):
                #     print(f"{i+1}: {msg}")
                new_msg = input("More?: ")
                if new_msg.strip() == "":
                    prior_requests = []  # reset conversation on empty input
                    print("Conversation reset.\n")
                    continue

            prior_requests.append(new_msg)  
            query = "\n Update: ".join([msg for msg in prior_requests])
            query += " Otherwise no changes."
        
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        try:
            if len(prior_requests) > 0:
                print("\n--- Continuing conversation with context ---\n")
                print("Previous conversation:" )   
                for i, msg in enumerate(prior_requests):
                    print(f"{i+1}: {msg[0:200]}")
            else:
                print("\n--- New conversation ---\n")
            
            # the dependencies are only available via RunContext in tools!
            # tools must be registered with takes_ctx=True to access deps
            updated_deps = AstroDependencies(
                # make sure these defaults are 'now' at runtime
                default_time=datetime.datetime.now(ZoneInfo("America/Chicago")).strftime("%H:%M"),
                default_date=datetime.datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d"),
                default_timezone="America/Chicago",
            )
            result = await astro_agent.run(query, deps=updated_deps)
                            
            print("FINAL Bot Output --->:")
            if isinstance(result.output, Plan):
                if len(result.output.dsos) == 0:
                    print("Result: No deep space objects found in the plan.")
                else:
                    print("Result Plan Deep Space Objects:")
                    for i, dso in enumerate(result.output.dsos):
                        print(f"Result[{i}]: {dso.name} ({dso.catalog}), Type: {dso.type}, Class: {dso.clasz}, Mag: {dso.vis_mag}, Size: {dso.size}, Constellation: {dso.constellation}, Altitude: {dso.altitude}, Azimuth: {dso.azimuth}")
                        #print(f"Result[{i}]: DSO ID: {dso.dso_id} -> {dso.info}")
                print(f"Equipment included in plan: {result.output.equipment}")
                print(f"Observer Context included in plan: {result.output.observer_context}")
            else:
                print(f"Output was NOT a PLAN: {result.output}")

            # conversation.append(result.new_messages())
            # prior_requests.append(new_msg)

            # print("\nusage:", result.usage(), "\n")
            # print("\n---------------------New messages:\n", result.new_messages(), "\n")
            # print("\nDebugging messages:")
            # print(result.all_messages(), "\n")

        except Exception as e:
            print(f"Agent throws Error: {e}\n")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
