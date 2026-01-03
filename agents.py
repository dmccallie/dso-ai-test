from dataclasses import dataclass
from pathlib import Path
from json import tool
from time import perf_counter
from typing import Optional
from zoneinfo import ZoneInfo
from pydantic import BaseModel, PositiveFloat, Field, create_model
from pydantic_ai import Agent, RunContext
import datetime

import dotenv

from ai_data_models import AstroDependencies, Camera, DeepSpaceObjectID, Plan, Telescope, Equipment, ObserverContext, DeepSpaceObject
from ai_data_models import EquipmentQuery, ObserverContextQuery
from ai_data_models import model_string

from ai_astronomy_utils import ai_localize_dso, convert_utc_iso_to_local

# from astro_test_plan import DeepSpaceObject

dotenv.load_dotenv()

# db_path = Path("./dso_data.db")

### QUALIIFYING DSO TOOL ###

# @astro_agent.tool
# RunContext[AstroDependencies] needs to be the first argument if takes_ctx=True
# apparently an output_type tool automaticaly gets takes_ctx=True???
async def return_dsos_observer_gear(ctx: RunContext[AstroDependencies], ai_query:str, observer_context: ObserverContext,
                                     gear: Equipment, ) -> Plan:
    import sqlite3

    """
    Tool: return_dsos_observer_gear(observer_context: ObserverContext, gear: Equipment) -> Plan
    - The main agent will call this tool after setting up observer context and gear and generating the SQL query.
    """
    print(f"[return_dsos_observer_gear] called with astrodependencies ctx:\n{ctx.deps}\n")
    print(f"[return_dsos_observer_gear] called with query:{ai_query},\n\t observer_context: {observer_context} and gear: {gear}")

    try:
        conn = sqlite3.connect(ctx.deps.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # create a temporary in-memory table we can map localizations into
        # make it match the DeepSpaceObject fields we want to fill in
        start = perf_counter()
        cursor.execute('''
            DROP TABLE IF EXISTS dso_localized;
        ''')
        
        cursor.execute("""
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
            rise_time TEXT, /* iso format full datetime string for utc */
            set_time TEXT, /* iso format full datetime string for utc */
            transit_time TEXT /* iso format full datetime string for utc */
        );
        """)
        
        # fetch all dso static info (all objects for now)
        cursor.execute("""
            SELECT * from dso;
        """)
        all_dsos = cursor.fetchall()

        # create an observation datetime from observer_context
        # combine date and time into one string and parse
        # we should never get here with missing date or time, but just in case use defaults
        assert observer_context is not None
        assert observer_context.observe_date is not None
        assert observer_context.observe_time is not None
        assert observer_context.timezone is not None

        local_date = observer_context.observe_date if observer_context.observe_date else ctx.deps.default_date
        local_time = observer_context.observe_time if observer_context.observe_time else ctx.deps.default_time
        local_tz = observer_context.timezone if observer_context.timezone else ctx.deps.default_timezone


        # ensure we have no seconds in time string (AI puts them there sometimes)
        if len(local_time.split(":")) == 3:
            local_time = ":".join(local_time.split(":")[0:2])

        date_iso = f"{local_date}T{local_time}"
        dt = datetime.datetime.strptime(f"{local_date} {local_time}", "%Y-%m-%d %H:%M")

        # attach timezone
        dt = dt.replace(tzinfo=ZoneInfo(local_tz))
        print(f"[get_qualifying_deep_space_objects] recreate full datetime = {dt.isoformat()} for ({local_tz})")
        
        # for each DSO, compute localization based on observer_context if present
        for dso_row in all_dsos:
            # default values
            altitude = None
            azimuth = None
            air_mass = None
            rise_time = None
            set_time = None
            transit_time = None

            # if observer_context has location compute localization
            # we already created a datetime object 'dt' above
            if observer_context.latitude_deg is not None and observer_context.longitude_deg is not None:

                altitude, azimuth, air_mass, visible, rise_time, transit_time, set_time = \
                  ai_localize_dso(dso_row['ra_dd'], dso_row['dec_dd'],
                     observer_context.latitude_deg, observer_context.longitude_deg,
                     dt.isoformat(), local_tz)
            
            # insert into temp table
            cursor.execute("""
                INSERT OR REPLACE INTO dso_localized (
                    dso_id, catalog, name, ra_dd, dec_dd, type, class, vis_mag, maj_axis, min_axis, size, constellation,
                    constellation_abbr, altitude, azimuth, air_mass, rise_time, set_time, transit_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                dso_row['dso_id'],
                dso_row['catalog'],
                dso_row['name'],
                dso_row['ra_dd'],
                dso_row['dec_dd'],
                dso_row['type'],
                dso_row['class'],
                dso_row['vis_mag'],
                dso_row['maj_axis'],
                dso_row['min_axis'],
                dso_row['size'],
                dso_row['constellation'],
                dso_row['constellation_abbr'],
                altitude,
                azimuth,
                air_mass,
                rise_time,
                set_time,
                transit_time,
            ))

        print(f"[return_dsos_observer_gear] localized {len(all_dsos)} DSOs - took {(perf_counter() - start)*1000:.3f} milliseconds")

        # do we need this?
        conn.commit()

        # now execute the provided query against the localized table
        cursor.execute(ai_query)
        rows = cursor.fetchall()

    except Exception as e:
        print("[return_dsos_observer_gear] SQL error:", repr(e))
        raise
    finally:
        conn.close()

    if not rows:
        print("[return_dsos_observer_gear] No results found.")
        return Plan(dsos=[], equipment=gear, observer_context=observer_context)
    
    objects = []
    for row in rows:
        #dso_id = row['dso_id']
        rise = convert_utc_iso_to_local(row['rise_time'], "America/Chicago") if row['rise_time'] is not None else "-" 
        set = convert_utc_iso_to_local(row['set_time'], "America/Chicago") if row['set_time'] is not None else "-"
        transit = convert_utc_iso_to_local(row['transit_time'], "America/Chicago") if row['transit_time'] is not None else "-"

        dso = DeepSpaceObject(
            dso_id=row['dso_id'],
            catalog=row['catalog'],
            name=row['name'],
            ra_dd=row['ra_dd'],
            dec_dd=row['dec_dd'],
            type=row['type'],
            clasz=row['class'],
            vis_mag=row['vis_mag'],
            maj_axis=row['maj_axis'] if row['maj_axis'] != "" else 0.0,
            min_axis=row['min_axis'] if row['min_axis'] != "" else 0.0,
            size=row['size'],
            constellation=row['constellation'],
            constellation_abbr=row['constellation_abbr'],
            altitude=row['altitude'],
            azimuth=row['azimuth'],
            air_mass=row['air_mass'],
            rise_time=rise,
            set_time=set,
            transit_time=transit,
        )
        objects.append(dso) 

    return Plan( 
        dsos=objects,
        equipment=gear,
        observer_context=observer_context)

#################################################
#### GEAR AGENT ####

gear_agent = Agent(
    model_string, 
    instructions=
        """
        - You are an agent designed find specific data about astronomy gear as used for visual observation and astrophotography.
        - When the user provides a nickname, common abbreviation, or product name for astronomy equipment,
            you should use the tools to find detailed specifications about telescopes and cameras.
        - Your goal is to find telescope focal length and f-ratio, and camera sensor size and pixel size.
        - If there is only information for telescope or camera, return just that part.
        - If you can only find partial information, return what you can.
        - Do not make up values.
        """
    ,
    deps_type=AstroDependencies,
    retries=2,
    output_type=Equipment,
)

# this seems to work when the agent needs this info BEFORE the defaults are accepted
#   since the agent needs to do some work to find FP, F-Ratio, sensor size, pixel size, etc
# instruction tools always get takes_ctx=True??
@gear_agent.instructions
async def custom_gear_instructions(ctx: RunContext[AstroDependencies]) -> str:
    return f"""
    - Use the following defaults if the user did not specify equipment, or you cannot find information about the requested gear:
        - default telescope: {ctx.deps.default_telescope}
        - default camera: {ctx.deps.default_camera}
    """

# just add this function to the main agent's tool list @gear_agent.tool_plain

# can't use @gear_agent.tool here, but can set it in the main agent tools list using Tool()
# ctx must be first argument if Tool(takes_ctx=True)
# test param foo gets seen but not listed in the logging trace??
async def infer_equipment_specs( ctx: RunContext[AstroDependencies], query: EquipmentQuery, foo:str="bar") -> Equipment:
    """
    Use AI or internet resources to infer detailed specifications for astronomy equipment based on EquipmentQuery, which wil
    contain descriptions or nicknames of the equipment extracted from the users overall request.
    """

    # If not found locally, use AI to infer specs
    try:
        print(f"[infer_equipment_specs] invoking gear_agent for AI inference with query: {query.text}")
        print(f"[infer_equipment_specs] called with context defaults: {ctx.deps}")
        print(f"[infer_equipment_specs] called with foo: {foo}")

        # defaultedEquipment = create_model(
        #     "Equipment",
        #     telescope=(Optional[Telescope], Field(default=Equipment().telescope,
        #         description="The telescope specifications, if available.")),
        #     camera=(Optional[Camera], Field(default=Equipment().camera,
        #         description="The camera specifications, if available.")),
        #         __base__=Equipment
        # )
        ai_result = await gear_agent.run(query.text, output_type=Equipment, deps=ctx.deps)
        print(f"[infer_equipment_specs] tool outputs inferred gear: {ai_result.output}")
        #return Equipment(**ai_result.output.model_dump())
        return ai_result.output
    
    except Exception as e:
        print(f"[infer_equipment_specs] AI inference failed for '{query.text}': {e}")
        return Equipment()  # Return empty Equipment on failure


################################################################
### Observer Context Agent and Tool ###

observer_context_agent = Agent(
    model_string, #'openai:gpt-5.1', 
    instructions=
        """
        - You are an agent designed find specific data about the observer's context for an astronomy observation session.
        - from the user's textual description, extract or infer the following fields into ObserverContext:
             location, latitude_deg, longitude_deg, observe_date, observe_time, and timezone.
        - Follow these rules when inferring the ObserverContext:
        - If the user provides a location, fetch the latitude and longitude if possible..
        - If the user provides a relative date like "tonight" or "this weekend", convert that to an actual date string using default_date from dependencies.
            for example, if default_date is "2024-06-15" and user says "tonight", use "2024-06-15" as the date string.
            If the user says "tomorrow" use "2024-06-16". Do not use UTC dates.
        - If the user provides an observation time like "22:00" capture that as observe_time.
        - If the user provides a relative time like "now" or "in 2 hours", convert that to actual time string using default_time from the defaults.
            for example, if the default_time is "14:30" and user says "in 2 hours", use "16:30" as the time string. Do not use UTC times.
        - Do not worry if the observe_time or observe_date are in the past.
        - If after considering details, any information that is missing or cannot be determined, use the defaults in the output structure.
        """
    ,
    retries=2,
    deps_type=AstroDependencies,
    output_type=ObserverContext,
)

@observer_context_agent.instructions
async def custom_instructions(ctx: RunContext[AstroDependencies]) -> str:
    return f"""
    - Use the following defaults when inferring missing date and time information:
        - default date: {ctx.deps.default_date}
        - default time: {ctx.deps.default_time}
        - default timezone: {ctx.deps.default_timezone}
    """


async def infer_observer_context( ctx:RunContext[AstroDependencies], q: ObserverContextQuery,) -> ObserverContext:
    """
    - Use AI to extract location, latitude, longitude, date, time, from a textual description provided by the user.
    - If any information is missing or cannot be determined, use defaults from AstroDependencies where appropriate.
    """
    try:
        print("[infer_observer_context]invoking observer_context_agent for context inference with description:", q.text)
        print(f"[infer_observer_context] called with context defaults: {ctx.deps}")
        myObserverContext = create_model(
            "ObserverContext",
            location=(str, Field(default=ctx.deps.default_location, description="The location where the user is observing from.")),
            latitude_deg=(Optional[float], Field(default=ctx.deps.default_latitude, description="The latitude in degrees.")),
            longitude_deg=(Optional[float], Field(default=ctx.deps.default_longitude, description="The longitude in degrees.")),
            observe_date=(Optional[str], Field(default=ctx.deps.default_date, description="The observation date in local terms.")),
            observe_time=(Optional[str], Field(default=ctx.deps.default_time, description="The observation time in local hours.")),
            timezone=(str, Field(default=ctx.deps.default_timezone, description="The IANA timezone string for the observation session.")),
            __base__=ObserverContext
        )
        ai_result = await observer_context_agent.run(q.text, deps=ctx.deps, output_type=myObserverContext)
        print(f"[infer_observer_context] tool outputs inferred context: {ai_result.output}")
        
        return ai_result.output
    
    except Exception as e:
        print(f"[infer_observer_context] AI inference failed for description '{q.text}': {e}")
        return ObserverContext(location="")  # Return minimal ObserverContext on failure
    
# add a tool to convert local date/time to utc iso format for storage
# from GPT-5.1
async def convert_local_day_and_time_to_utc_iso(local_date: str, local_time: str, timezone_str: str) -> str:
    """
    Convert a local date/time in a given IANA timezone to a UTC ISO-8601 string.

    Example:
        convert_local_day_and_time_to_utc_iso("2025-12-04", "22:00", "America/Chicago")
        -> "2025-12-05T04:00:00Z"
    """
    # parse local date + time to naive datetime
    dt_str = f"{local_date} {local_time}"
    for time_fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            naive = datetime.datetime.strptime(dt_str, time_fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(
            f"Invalid date/time format: date='{local_date}', time='{local_time}'"
        )

    # attach timezone (treating this as wall-clock time in that zone)
    tz = ZoneInfo(timezone_str)
    local_dt = naive.replace(tzinfo=tz)

    # convert to UTC
    utc_dt = local_dt.astimezone(ZoneInfo("UTC"))

    # return canonical UTC ISO-8601 with 'Z'
    return utc_dt.replace(tzinfo=None).isoformat(timespec="seconds") + "Z"

