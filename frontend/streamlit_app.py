import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Literal, Optional, Union

import nest_asyncio
import streamlit as st
from pydantic import Field, create_model
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

nest_asyncio.apply()
st.set_page_config(page_title="Neuronworks Travel Agent", page_icon="✈️", layout="wide")

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root{--bg:#080b14;--blue:#2563eb;--violet:#7c3aed;--border:rgba(255,255,255,.09);--text:#e5e7eb;--muted:#94a3b8}
    .stApp{background:radial-gradient(circle at 10% 0%,rgba(37,99,235,.55) 0,transparent 35%),radial-gradient(circle at 90% 10%,rgba(124,58,237,.50) 0,transparent 32%),var(--bg)}
    .block-container{max-width:1180px;padding-top:4rem;padding-bottom:7rem}
    .hero{padding:28px 30px;border-radius:24px;margin-bottom:22px;background:linear-gradient(135deg,rgba(37,99,235,.30),rgba(124,58,237,.25));border:1px solid rgba(255,255,255,.12);box-shadow:0 18px 60px rgba(0,0,0,.25)}
    .hero h1{margin:0;color:#fff;font-size:2.25rem;letter-spacing:-.04em}.hero p{margin:8px 0 0;color:#cbd5e1}
    .badge{display:inline-block;padding:5px 11px;border-radius:999px;background:rgba(255,255,255,.10);color:#e2e8f0;font-size:.78rem;border:1px solid rgba(255,255,255,.12)}
    section[data-testid="stSidebar"]{background:rgba(8,11,20,.96);border-right:1px solid var(--border)}
    div[data-testid="stChatMessage"]{border:1px solid var(--border);border-radius:20px;padding:1.15rem 1.25rem;margin:.8rem 0;background:rgba(15,23,42,.80);box-shadow:0 10px 30px rgba(0,0,0,.18)}
    div[data-testid="stChatMessageContent"]{color:var(--text)}
    div[data-testid="stChatMessageContent"] h1,div[data-testid="stChatMessageContent"] h2,div[data-testid="stChatMessageContent"] h3{color:#fff}
    div[data-testid="stExpander"]{border:1px solid var(--border);border-radius:14px}.muted{color:var(--muted);font-size:.82rem}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="badge">● LIVE MCP TRAVEL INTELLIGENCE</div>
      <h1>✈️ Neuronworks Travel Agent</h1>
      <p>Flights · Hotels · Places · Restaurants · Weather · Currency · Budget</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    server_url = st.text_input(
        "MCP Server URL",
        value="https://neuronworks-travel-agent.onrender.com/sse",
    )
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.text_input(
        "Groq API Key",
        type="password",
    )
    nvidia_api_key = os.environ.get("NVIDIA_API_KEY") or st.text_input(
        "NVIDIA API Key",
        type="password",
    )
    if not groq_api_key:
        st.warning("Enter GROQ_API_KEY for fast request extraction / routing.")
        st.stop()
    if not nvidia_api_key:
        st.warning("Enter NVIDIA_API_KEY for Llama 3.3 70B itinerary planning.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key
    st.success("🟢 Connected")
    st.markdown(
        '<div class="muted">Fast parser/router: GPT-OSS 20B · Planner: Llama 3.3 70B</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# MCP schema helpers
# ---------------------------------------------------------------------
def schema_type(schema: Dict[str, Any]):
    if not isinstance(schema, dict):
        return Any
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        try:
            return Literal[tuple(schema["enum"])]
        except TypeError:
            pass
    t = schema.get("type")
    if t == "string":
        return str
    if t == "number":
        return float
    if t == "integer":
        return int
    if t == "boolean":
        return bool
    if t == "array":
        return List[schema_type(schema.get("items", {}))]
    if t == "object":
        return Dict[str, Any]
    any_of = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(any_of, list):
        types = [schema_type(x) for x in any_of if x.get("type") != "null"]
        if len(types) == 1:
            return Optional[types[0]]
        if types:
            return Union[tuple(types)]
    return Any


def create_pydantic_model_from_schema(name: str, schema: Dict[str, Any]):
    fields = {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
    for field_name, info in properties.items():
        annotation = schema_type(info)
        default = info.get("default", ... if field_name in required else None)
        if field_name not in required and default is None:
            annotation = Optional[annotation]
        fields[field_name] = (
            annotation,
            Field(default=default, description=info.get("description", "")),
        )
    return create_model(f"{name}Input", **fields)


def as_list(value):
    return value if isinstance(value, list) else []


def as_dict(value):
    return value if isinstance(value, dict) else {}


def parse_iso_date(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None


def money(value, currency="USD"):
    try:
        return f"{currency} {float(value):,.2f}"
    except Exception:
        return "Unavailable"


# ---------------------------------------------------------------------
# Data-quality filters
# ---------------------------------------------------------------------
def clean_attractions(items):
    name_deny = re.compile(
        r"\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|"
        r"bus\s*station|railway|parking|signal|flyover|underpass|bypass|"
        r"overpass|salai|theru|sandhu|mawatha|marg|nagar|colony|layout|"
        r"township|extension|ward|sector|block|circle|chowk|hospital)\b",
        re.I,
    )
    category_deny = (
        "administrative",
        "populated_place",
        "residential",
        "postcode",
        "suburb",
        "neighbourhood",
        "neighborhood",
        "locality",
        "commercial.building",
        "office",
        "hospital",
    )
    output, seen = [], set()
    for item in as_list(items):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen or name_deny.search(name):
            continue
        categories = [str(x).lower() for x in item.get("categories", [])]
        if any(any(bad in c for bad in category_deny) for c in categories):
            continue
        allowed = any(
            c.startswith("tourism.")
            or "museum" in c
            or "culture" in c
            or "place_of_worship" in c
            or "historic" in c
            or c.startswith("natural")
            or "park" in c
            or "heritage" in c
            for c in categories
        )
        if not allowed:
            continue
        seen.add(key)
        output.append(item)
    return output


def clean_restaurants(items):
    name_deny = re.compile(
        r"\b(street|road|lane|mawatha|marg|salai|theru|sandhu|highway|"
        r"junction|bus\s*stop|station|nagar|colony)\b",
        re.I,
    )
    output, seen = [], set()
    for item in as_list(items):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name_deny.search(name):
            continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen:
            continue
        categories = [str(x).lower() for x in item.get("categories", [])]
        if not any(c.startswith("catering.") for c in categories):
            continue
        seen.add(key)
        output.append(item)
    return output


# ---------------------------------------------------------------------
# Deterministic answer renderer
# ---------------------------------------------------------------------
def render_trip(trip, plan):
    request = trip.get("request", {})
    services = trip.get("services", {})
    live = trip.get("liveDataSummary", {})

    origin = request.get("origin", "")
    city = request.get("destinationCity", "")
    country = request.get("destinationCountry", "")
    depart = str(request.get("departDate", ""))
    return_date = str(request.get("returnDate", ""))
    travelers = request.get("travelers", 1)
    nights = int(request.get("durationNights") or 0)
    days = int(request.get("calendarDays") or nights + 1)

    flights = as_list(services.get("flights"))
    hotels = as_list(services.get("hotels"))
    attractions = clean_attractions(services.get("attractions"))
    restaurants = clean_restaurants(services.get("restaurants"))
    weather = as_dict(services.get("weather"))
    budget = as_dict(services.get("budget"))

    lines = [
        "## ✈️ Trip at a glance\n",
        f"**{origin} → {city}, {country}**  ",
        f"**{depart} → {return_date} · {travelers} traveler(s) · {nights} night(s) / {days} calendar day(s)**  ",
        f"Budget level: **{request.get('budgetLevel', 'budget')}**\n",
    ]

    lines += ["## 🛫 Flights\n"]
    if flights:
        lines.append("| Airline | Price | Departure | Arrival | Duration | Stops |\n|---|---:|---|---|---:|---:|")
        for f in flights[:5]:
            stops = int(f.get("stops", 0) or 0)
            lines.append(
                f"| {f.get('airline', 'Unknown')} | {money(f.get('price'), f.get('currency', 'USD'))} | "
                f"{f.get('departure', '—')} | {f.get('arrival', '—')} | {f.get('duration', '—')} | "
                f"{'Non-stop' if stops == 0 else f'{stops} stop(s)'} |"
            )
        lines.append("\n*Live provider results; prices and availability can change.*\n")
    else:
        lines.append(
            f"**Live flights unavailable.** {as_dict(services.get('flights')).get('error', 'No live flight options were returned.')}\n"
        )

    lines += ["## 🏨 Hotels\n"]
    if hotels:
        lines.append("| Hotel | Nightly | Rating | Reviews |\n|---|---:|---:|---:|")
        for h in hotels[:6]:
            rating = f"{float(h['rating']):.1f}" if isinstance(h.get("rating"), (int, float)) else "—"
            lines.append(
                f"| {h.get('name', 'Unknown')} | {money(h.get('price'), h.get('currency', 'USD'))} | "
                f"{rating} | {h.get('reviews', '—')} |"
            )
        lines.append(f"\n*Live hotel rates returned for {depart} → {return_date}.*\n")
    else:
        lines.append(
            f"**Live hotels unavailable.** {as_dict(services.get('hotels')).get('error', 'No live hotel options were returned.')}\n"
        )

    lines += ["## 📍 Things to do\n"]
    if attractions:
        used = []
        for day in plan.get("days", []):
            used.extend(day.get("attraction_ids", []))
        for idx in dict.fromkeys(used):
            if 0 <= idx < len(attractions):
                p = attractions[idx]
                lines.append(
                    f"- **{p.get('name')}**"
                    + (f" — {p.get('description')}" if p.get("description") else "")
                )
        if not used:
            lines.append("No attractions were selected from the verified results.")
    else:
        lines.append("No high-quality live tourist attractions were returned. I won't pad the itinerary with roads or random map objects.")
    lines.append("")

    lines += ["## 🍽️ Food picks\n"]
    if restaurants:
        for r in restaurants[:8]:
            lines.append(
                f"- **{r.get('name')}**"
                + (f" — {r.get('description')}" if r.get("description") else "")
            )
        lines.append("\n*Recommendations only; no reservation is implied.*\n")
    else:
        lines.append("No verified restaurant results were returned.\n")

    lines += ["## 🌦️ Weather\n"]
    weather_rows = as_list(weather.get("results"))
    if weather_rows:
        lines.append("| Date | Temp | Feels like | Conditions | Humidity | Rain |\n|---|---:|---:|---|---:|---:|")
        for w in weather_rows:
            lines.append(
                f"| {w.get('date', '—')} | {w.get('temperature', '—')}°C | "
                f"{w.get('feelsLike', '—')}°C | {w.get('description', '—')} | "
                f"{w.get('humidity', '—')}% | {w.get('precipitationProbability', '—')}% |"
            )
        coverage = weather.get("coverage", {})
        if coverage:
            lines.append(
                f"\n*Live forecast coverage: **{coverage.get('returnedStart')} → {coverage.get('returnedEnd')}**. No weather is extrapolated.*\n"
            )
    else:
        lines.append(
            f"**Live weather unavailable.** {weather.get('error', 'No forecast data was returned for the requested dates.')}\n"
        )

    lines += ["## 💰 Budget\n"]
    if budget:
        currency = budget.get("currency", "USD")
        breakdown = budget.get("breakdown", {})
        lines.append(f"**Generic estimate:** {money(budget.get('total_budget'), currency)}")
        lines.append(f"- Flights estimate: {money(breakdown.get('flights_estimate'), currency)}")
        lines.append(f"- Accommodation estimate: {money(breakdown.get('accommodation_estimate'), currency)}")
        lines.append(f"- Daily expenses estimate: {money(breakdown.get('daily_expenses_estimate'), currency)}")
        lines.append("\n*Generic planning estimate only — not a live booking total.*")
    if live.get("complete"):
        lines.append(f"\n**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'), live.get('currency', 'USD'))}")
        lines.append("Includes cheapest returned live flight + cheapest returned hotel nightly rate × nights; excludes food, local transport, activities and unreturned fees/taxes.")
    else:
        lines.append("\n**Live-data subtotal:** incomplete because a usable live flight or hotel price was missing.")
    lines.append("")

    lines += ["## 🗓️ Suggested itinerary\n"]
    if not plan.get("days"):
        lines.append("I don't have enough verified attraction data to create a responsible day-by-day itinerary without inventing places.")
    else:
        for i, day in enumerate(plan["days"], 1):
            lines.append(f"### Day {i} · {day.get('date')} · {day.get('title')}")
            if i == 1:
                lines.append("- ✈️ Arrival / check-in")
            for j, idx in enumerate(day.get("attraction_ids", [])):
                if 0 <= idx < len(attractions):
                    lines.append(f"- **{'Morning' if j == 0 else 'Afternoon'}:** {attractions[idx].get('name')}")
            rid = day.get("restaurant_id")
            if rid is not None and 0 <= rid < len(restaurants):
                lines.append(f"- 🍽️ **Food:** {restaurants[rid].get('name')}")
            if day.get("reason"):
                lines.append(f"- _Why this works:_ {day.get('reason')}")
            if i == len(plan["days"]):
                lines.append("- 🧳 Check-out / departure")
            if not day.get("attraction_ids") and rid is None:
                lines.append("- Keep this period flexible rather than inventing another attraction.")
            lines.append("")

    lines += [
        "## ⚠️ Notes\n",
        "- Live prices and availability can change before booking.",
        "- Generic budget estimates are not live booking totals.",
        "- Weather is shown only for dates covered by the live forecast provider.",
        "- The itinerary uses only provider-returned attraction and restaurant IDs.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Fast router / freshness classifier
# ---------------------------------------------------------------------
async def needs_fresh_tool_data(user_msg, trip_data, last_answer):
    if not trip_data:
        return True
    prompt = f"""Classify exactly REUSE or FRESH.
Existing data keys: {', '.join(trip_data.keys())}
Last answer: {last_answer[:500]}
New user message: {user_msg}
REUSE only if completely answerable from existing trip data.
FRESH for a new destination, dates, origin, travelers, category, or live information.
Reply exactly one word."""
    try:
        router = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=5)
        result = await router.ainvoke([HumanMessage(content=prompt)])
        return not (result.content or "").strip().upper().startswith("REUSE")
    except Exception:
        return True


def current_trip_is_valid(trip):
    request = trip.get("request", {})
    start = parse_iso_date(request.get("departDate"))
    end = parse_iso_date(request.get("returnDate"))
    today = date.today()
    if not start or not end:
        return False, "Please provide valid departure and return dates in YYYY-MM-DD format."
    if start < today:
        return False, f"The departure date {start.isoformat()} is in the past. Today is {today.isoformat()}."
    if end <= start:
        return False, "The return date must be after the departure date."
    return True, ""


# ---------------------------------------------------------------------
# ONE 70B CALL: itinerary planning only
# ---------------------------------------------------------------------
def extract_json(text: str):
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
    if fenced:
        return json.loads(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Planner did not return JSON")
    return json.loads(text[start:end + 1])


def build_planner_payload(trip, user_request):
    services = trip.get("services", {})
    request = trip.get("request", {})
    attractions = clean_attractions(services.get("attractions"))
    restaurants = clean_restaurants(services.get("restaurants"))
    return {
        "user_request": user_request,
        "trip": {
            "origin": request.get("origin"),
            "destination_city": request.get("destinationCity"),
            "destination_country": request.get("destinationCountry"),
            "departure_date": request.get("departDate"),
            "return_date": request.get("returnDate"),
            "travelers": request.get("travelers"),
            "nights": request.get("durationNights"),
            "budget_level": request.get("budgetLevel"),
        },
        "verified_attractions": [
            {"id": i, "name": x.get("name"), "description": x.get("description"), "address": x.get("address"), "categories": x.get("categories", [])}
            for i, x in enumerate(attractions[:16])
        ],
        "verified_restaurants": [
            {"id": i, "name": x.get("name"), "description": x.get("description"), "address": x.get("address"), "categories": x.get("categories", [])}
            for i, x in enumerate(restaurants[:12])
        ],
        "live_flights": [
            {"airline": f.get("airline"), "price": f.get("price"), "currency": f.get("currency"), "departure": f.get("departure"), "arrival": f.get("arrival"), "duration": f.get("duration"), "stops": f.get("stops")}
            for f in as_list(services.get("flights"))[:8]
        ],
        "live_hotels": [
            {"name": h.get("name"), "price": h.get("price"), "currency": h.get("currency"), "rating": h.get("rating"), "reviews": h.get("reviews"), "address": h.get("address")}
            for h in as_list(services.get("hotels"))[:8]
        ],
        "weather": as_dict(services.get("weather")),
        "budget": as_dict(services.get("budget")),
    }


def validate_plan(plan, attractions, restaurants, start, end):
    if not isinstance(plan, dict) or not isinstance(plan.get("days"), list):
        raise ValueError("Invalid planner JSON")
    valid_dates = set()
    current = start
    while current <= end:
        valid_dates.add(current.isoformat())
        current += timedelta(days=1)

    seen = set()
    normalized = []
    for day in plan["days"]:
        if not isinstance(day, dict) or day.get("date") not in valid_dates:
            continue
        ids = []
        for raw in (day.get("attraction_ids") or [])[:2]:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(attractions) and idx not in seen:
                ids.append(idx)
                seen.add(idx)
        rid = day.get("restaurant_id")
        try:
            rid = int(rid) if rid is not None else None
        except (TypeError, ValueError):
            rid = None
        if rid is not None and not (0 <= rid < len(restaurants)):
            rid = None
        normalized.append({
            "date": day["date"],
            "title": str(day.get("title") or "Travel day").strip(),
            "attraction_ids": ids,
            "restaurant_id": rid,
            "reason": str(day.get("reason") or "").strip(),
        })
    normalized.sort(key=lambda x: x["date"])
    return {"days": normalized}


def fallback_plan(attractions, restaurants, start, end):
    days = []
    current = start
    attraction_index = 0
    restaurant_index = 0
    while current <= end:
        is_first = current == start
        is_last = current == end
        selected = [] if is_first or is_last else ([attraction_index] if attraction_index < len(attractions) else [])
        if selected:
            attraction_index += 1
        days.append({
            "date": current.isoformat(),
            "title": "Arrival" if is_first else ("Departure" if is_last else "Sightseeing"),
            "attraction_ids": selected,
            "restaurant_id": restaurant_index if restaurants else None,
            "reason": "Safe fallback using only verified live POIs.",
        })
        if restaurants:
            restaurant_index = (restaurant_index + 1) % len(restaurants)
        current += timedelta(days=1)
    return {"days": days}


async def build_itinerary_with_nvidia(llm, trip, user_request):
    request = trip.get("request", {})
    start = parse_iso_date(request.get("departDate"))
    end = parse_iso_date(request.get("returnDate"))
    attractions = clean_attractions(trip.get("services", {}).get("attractions"))
    restaurants = clean_restaurants(trip.get("services", {}).get("restaurants"))
    if not start or not end:
        return {"days": []}

    payload = build_planner_payload(trip, user_request)
    prompt = f"""You are the itinerary planner inside a travel agent.
Choose and order ONLY from VERIFIED provider data below.
Return JSON ONLY. No Markdown. No prose outside JSON.

Schema:
{{"days":[{{"date":"YYYY-MM-DD","title":"short title","attraction_ids":[0,1],"restaurant_id":0,"reason":"one short sentence"}}]}}

Rules:
- Use only supplied IDs.
- Never invent places, restaurants, activities, prices or weather.
- Dates must stay within the trip.
- Each attraction at most once.
- 0–2 attractions per day.
- Arrival and departure days should be light.
- Never use roads, streets, paths, hospitals, administrative POIs, random statues or weak map infrastructure as attractions.
- Prefer major historic, cultural, religious, museum or nature sites.
- Leave days light when verified data is insufficient.
- Restaurant IDs must come only from verified restaurants.
- Keep each day geographically sensible using the returned addresses/descriptions where possible.

USER REQUEST:
{user_request}

VERIFIED DATA:
{json.dumps(payload, ensure_ascii=False, indent=2)}"""
    try:
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        return validate_plan(extract_json(result.content), attractions, restaurants, start, end)
    except Exception:
        return fallback_plan(attractions, restaurants, start, end)


# ---------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------
async def run_agent(chat_history, trip_data, chat_container):
    async with AsyncExitStack() as stack:
        status = chat_container.empty()
        status.info("🔌 Connecting to MCP server…")
        try:
            transport = await stack.enter_async_context(sse_client(server_url))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            status.info("🧰 Loading travel services…")
            mcp_tools = await session.list_tools()

            tools = []
            for tool in mcp_tools.tools:
                async def call_mcp_tool(tool_name=tool.name, **kwargs):
                    return await session.call_tool(tool_name, arguments=kwargs)

                tools.append(
                    StructuredTool.from_function(
                        func=None,
                        coroutine=call_mcp_tool,
                        name=tool.name,
                        description=tool.description or "Travel MCP tool",
                        args_schema=create_pydantic_model_from_schema(tool.name, tool.input_schema),
                    )
                )

            # NVIDIA is now used ONLY for itinerary generation. This removes
            # the slow 70B tool-extraction call that was keeping the UI stuck
            # on "Extracting trip request…".
            nvidia_llm = ChatNVIDIA(
                model="meta/llama-3.3-70b-instruct",
                api_key=os.environ["NVIDIA_API_KEY"],
                temperature=0.0,
                top_p=0.7,
                max_completion_tokens=700,
                timeout=300,
            )

            latest_user = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), "")
            last_answer = next((m["content"] for m in reversed(chat_history) if m["role"] == "assistant"), "")
            fresh = await needs_fresh_tool_data(latest_user, trip_data, last_answer)

            if fresh:
                trip_data.clear()
                status.info("⚡ Fast request extraction with GPT-OSS 20B…")

                bundle_tool = next((t for t in tools if t.name == "build_trip_data"), None)
                if not bundle_tool:
                    raise RuntimeError("MCP server does not expose build_trip_data.")

                # Use the already-configured fast Groq model for argument
                # extraction. The expensive 70B model is reserved for the
                # actual itinerary reasoning step.
                extractor = ChatGroq(
                    model="openai/gpt-oss-20b",
                    temperature=0,
                    max_tokens=256,
                ).bind_tools([bundle_tool])

                first = await extractor.ainvoke(
                    [
                        HumanMessage(
                            content=(
                                "Extract the travel request and call build_trip_data. "
                                "Do not answer conversationally. "
                                "Never invent dates. If dates/origin/destination/travelers "
                                "are missing, leave them unresolved rather than guessing.\n\n"
                                f"USER REQUEST:\n{latest_user}"
                            )
                        )
                    ]
                )

                call = next(
                    (c for c in (first.tool_calls or []) if c.get("name") == "build_trip_data"),
                    None,
                )

                if not call:
                    status.empty()
                    return "## 🧭 I need a little more information\n\nPlease provide the origin, destination, departure date, return date, and number of travelers."

                args = dict(call.get("args") or {})
                start = parse_iso_date(args.get("departDate"))
                end = parse_iso_date(args.get("returnDate"))
                today = date.today()

                if not start or not end:
                    status.empty()
                    return "## ⚠️ Invalid dates\n\nPlease provide departure and return dates in YYYY-MM-DD format."
                if start < today:
                    status.empty()
                    return f"## ⚠️ Past travel date\n\nThe departure date **{start.isoformat()}** is in the past. Today is **{today.isoformat()}**."
                if end <= start:
                    status.empty()
                    return "## ⚠️ Invalid trip dates\n\nThe return date must be after the departure date."

                with st.expander("🧰 Live services used", expanded=False):
                    st.caption("Flights · Hotels · Attractions · Restaurants · Weather · Budget")
                    st.json(args)

                status.info("⚡ Fetching live trip data…")
                result = await bundle_tool.coroutine(**args)
                trip = json.loads(result.content[0].text)
                trip_data["build_trip_data"] = trip

                valid, error = current_trip_is_valid(trip)
                if not valid:
                    status.empty()
                    return f"## ⚠️ I can't plan this trip yet\n\n**{error}**"

                status.info("🧠 Planning your itinerary with Llama 3.3 70B…")
                plan = await build_itinerary_with_nvidia(nvidia_llm, trip, latest_user)
                status.empty()
                return render_trip(trip, plan)

            existing = trip_data.get("build_trip_data")
            if isinstance(existing, dict):
                valid, error = current_trip_is_valid(existing)
                if not valid:
                    status.empty()
                    return f"## ⚠️ Current trip is no longer valid\n\n**{error}**"
                status.info("♻️ Reusing verified live trip data…")
                plan = await build_itinerary_with_nvidia(nvidia_llm, existing, latest_user)
                status.empty()
                return render_trip(existing, plan)

            status.empty()
            return "I don't have verified live trip data for this session yet. Please submit a fresh trip request."

        except Exception as exc:
            status.error(f"Something went wrong: {exc}")
            return None


# ---------------------------------------------------------------------
# Chat UI
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trip_data" not in st.session_state:
    st.session_state.trip_data = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Where are you going? Try: Chennai → Madurai, Aug 20–25, 1 traveler")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = asyncio.run(
            run_agent(
                st.session_state.messages,
                st.session_state.trip_data,
                st.empty(),
            )
        )
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
