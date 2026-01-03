from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic import BaseModel, Field
from typing import Union
from pathlib import Path
    
# fetch openai api key from env file
import dotenv
dotenv.load_dotenv()

db_path = Path("./dso_data.db")

# define a deep space object, which will generally be the focus of an amateur astronomy session
class DeepSpaceObject(BaseModel):
    dso_id: str = Field(..., description="The unique identifier of the deep space object")
    name: str = Field(..., description="The most common name of the deep space object")
    clasz: str = Field(..., description=
        """
        The class of deep space object.
          We use 'clasz' to avoid python reserved keyword.
          We use 'clasz' for high-level classification and 'type' for more specific types.
        Our database uses the following classes (name, and db abbreviation):
        - Galaxy (Gal)
        - Nebula (Neb)
        - Cluster (Cls)
        - DS (Double Star)
        - Other (OTH)
        """
    )
    type: str = Field(..., description=
        """
        A more specific type of the deep space object, more specific than 'clasz'.
          We use (name, and db abbreviation):
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
        """
    )
    constellation: str = Field(..., description="The constellation in which the object is located")
    vis_mag: float = Field(..., description="The apparent visual magnitude (brightness) of the object")
    maj_axis: float = Field(..., description="The size of the major axis in arcminutes")
    min_axis: float = Field(..., description="The size of the minor axis in arcminutes")
    size: str = Field(..., description="A textual size description of the object, e.g. '5x10' or just '10'. Units are arcminutes.")
    ra_dd: float = Field(..., description="The right ascension in decimal degrees")
    dec_dd: float = Field(..., description="The declination in decimal degrees")
    catalog: str = Field(..., description=
                         """
                         The primary catalog designation of the object.
                         E.g., 'M 31', 'NGC 1976', 'Caldwell14', etc.
                         The catalog prefix should be included.
                         The most common catalog names are:
                         - Messier (M)
                         - New General Catalog (NGC)
                         - Index Catalog (IC)
                         - Caldwell (Caldwell)
                         - SH2 (Sharpless)
                         """)
    azimuth: float | None = Field(default=None, description="The azimuth in degrees (localized at session time)")
    altitude: float | None = Field(default=None, description="The altitude in degrees (localized at session time)")


astro_agent = Agent(
    "openai:gpt-5.1", #groq:llama-3.3-70b-versatile",
    instructions="""
    You are a friendly expert helping the user plan amateur astronomy sessions.

    When the user asks about deep-sky objects, you should:
    1. Use the tools to fetch data.
    1b. Use only the deep space object database for deep sky object information. Do not include other sources.
    2. If the result is a LIST of deep space objects, return as a list of DeepSpaceObject models. If the result is a COUNT, return as an integer.

        Table schema:
        CREATE TABLE dso (
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
            search_name TEXT
        )
        Note that "catalog" contains the catalog prefix as well as the catalog number or name, e.g., "M 31", "NGC 1976", etc.
        So to fetch all Messier objects, the query would be: SELECT * FROM dso WHERE catalog LIKE 'M %';
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
    3. If the user asks for a count of objects, return an integer count.

    """,
    output_type=[list[DeepSpaceObject], int],
    retries=2,
)


@astro_agent.tool_plain
async def get_deep_space_objects(query: str) -> list[DeepSpaceObject]:
    """
    Tool: get_deep_space_objects(query: str) -> list[DeepSpaceObject]

    - Fetch deep space objects from the SQLite database using the provided SQL query.
    - The `query` argument MUST be a valid SQLite SQL SELECT statement.
    - Do NOT include explanations or comments in `query`.
    - Example: "SELECT * FROM dso WHERE catalog LIKE 'M %';"

    """
    import sqlite3

    print(f"[get_deep_space_objects] called with query: {query}")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(query)
        rows = cursor.fetchall()
    except Exception as e:
        print("[get_deep_space_objects] SQL error:", repr(e))
        raise
    finally:
        conn.close()

    if not rows:
        print("[get_deep_space_objects] No results found.")
        return []
    
    objects = []
    for row in rows:
        # print(dict(row))  # Debug: print the row as a dictionary
        obj = DeepSpaceObject(
            dso_id=row['dso_id'],
            name=row['name'],
            clasz=row['class'],
            type=row['type'],
            constellation=row['constellation'],
            vis_mag=row['vis_mag'],
            size=row['size'],
            maj_axis=float(row['maj_axis']) if row['maj_axis'] != "" else 0.0,
            min_axis=float(row['min_axis']) if row['min_axis'] != "" else 0.0,
            ra_dd=float(row['ra_dd']) if row['ra_dd'] is not None else 0.0,
            dec_dd=float(row['dec_dd']) if row['dec_dd'] is not None else 0.0,
            catalog=row['catalog'],
            altitude=None,
            azimuth=None,
        )
        objects.append(obj)

    return objects

@astro_agent.tool_plain
async def count_deep_space_objects(query: str) -> int:
    """
        Count deep space objects using the query string and a sqlite database.
        The AI will generate the query string based on user input.
    """
    import sqlite3

    print(f"[count_deep_space_objects] called with query: {query}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row # Enable named column access
    cursor = conn.cursor()

    cursor.execute(query)
    row = cursor.fetchone()

    count = row[0] if row else 0

    print(f"[count_deep_space_objects] count: {count}")

    conn.close()
    return count

async def main() -> None:
    """Run a tiny REPL so you can test astro-agent."""

    import asyncio

    print("=== astro-agent demo ===")
    print("Type 'exit' or Ctrl+C to quit.\n")

    while True:
        try:
            user_msg = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        try:
            result = await astro_agent.run(user_msg,)
            print("Bot Output --->:")
            if isinstance(result.output, list):
                if len(result.output) == 0:
                    print("Result: No deep space objects found.")
                else:
                    for i, dso in enumerate(result.output):
                        print(f"Result[{i}]: {dso.name} ({dso.catalog}), Type: {dso.type}, Class: {dso.clasz}, Mag: {dso.vis_mag}, Size: {dso.size}, Constellation: {dso.constellation}")
            else:
                if isinstance(result.output, int):
                    print(f"Result: Count = {result.output}")
                else:
                    print(f"Result: {result.output}")

            print("\nusage:", result.usage(), "\n")
            # print("All messages:", result.all_messages(), "\n")
            print("\nDebugging messages:")
            print(result.all_messages(), "\n")

        except Exception as e:
            print(f"Agent throws Error: {e}\n")

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
