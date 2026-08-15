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
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool

nest_asyncio.apply()

import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Neuronworks Travel Agent",
    page_icon="✈️",
    layout="wide",
)

# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------
def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-base: #080b14;
            --blue: #2563eb;
            --violet: #7c3aed;
            --border-soft: rgba(255, 255, 255, 0.09);
            --border-strong: rgba(255, 255, 255, 0.12);
            --text-primary: #ffffff;
            --text-secondary: #e5e7eb;
            --text-muted: #94a3b8;
            --code-accent: #c4b5fd;
        }

        /* Page background */
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, var(--blue) 0, transparent 35%),
                radial-gradient(circle at 90% 10%, var(--violet) 0, transparent 32%),
                var(--bg-base);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 4.5rem;
            padding-bottom: 7rem;
        }

        /* Hero banner */
        .hero {
            padding: 28px 30px;
            border-radius: 24px;
            margin-bottom: 22px;
            background: linear-gradient(
                135deg,
                rgba(37, 99, 235, 0.30),
                rgba(124, 58, 237, 0.25)
            );
            border: 1px solid var(--border-strong);
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.25);
        }
        .hero h1 {
            margin: 0;
            color: var(--text-primary);
            font-size: 2.25rem;
            letter-spacing: -0.04em;
        }
        .hero p {
            margin: 8px 0 0;
            color: #cbd5e1;
            font-size: 1rem;
        }

        .status-pill {
            display: inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.10);
            color: #e2e8f0;
            font-size: 0.78rem;
            border: 1px solid var(--border-strong);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(8, 11, 20, 0.96);
            border-right: 1px solid var(--border-soft);
        }
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #f8fafc;
        }

        /* Chat messages */
        div[data-testid="stChatMessage"] {
            border: 1px solid var(--border-soft);
            border-radius: 20px;
            padding: 1.15rem 1.25rem;
            margin: 0.8rem 0;
            background: rgba(15, 23, 42, 0.76);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stChatMessageContent"] {
            color: var(--text-secondary);
        }
        div[data-testid="stChatMessageContent"] h1,
        div[data-testid="stChatMessageContent"] h2,
        div[data-testid="stChatMessageContent"] h3 {
            color: var(--text-primary);
            margin-top: 0.5rem;
        }
        div[data-testid="stChatMessageContent"] table {
            border-radius: 12px;
            overflow: hidden;
        }
        div[data-testid="stChatMessageContent"] code {
            color: var(--code-accent);
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            border: 1px solid var(--border-soft);
            border-radius: 14px;
        }

        .small-muted {
            color: var(--text-muted);
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="status-pill">● LIVE MCP TRAVEL INTELLIGENCE</div>
          <h1>✈️ Neuronworks Travel Agent</h1>
          <p>Flights · Hotels · Places · Restaurants · Weather · Currency · Budget</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


apply_theme()
render_hero()

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    server_url = st.text_input("MCP Server URL", value="https://neuronworks-travel-agent.onrender.com/sse")
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    hf_token = os.environ.get("HF_TOKEN") or st.text_input("Hugging Face Token", type="password")
    if not groq_api_key:
        st.warning("Enter GROQ_API_KEY for the semantic router.")
        st.stop()
    if not hf_token:
        st.warning("Enter HF_TOKEN for Llama 3.3 70B.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["HF_TOKEN"] = hf_token
    st.success("🟢 Connected")
    st.markdown('<div class="muted">Planner: Llama 3.3 70B Instruct · Router: GPT-OSS 20B</div>', unsafe_allow_html=True)


def schema_type(schema: Dict[str, Any]):
    if not isinstance(schema, dict): return Any
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        try: return Literal[tuple(schema["enum"])]
        except TypeError: pass
    t = schema.get("type")
    if t == "string": return str
    if t == "number": return float
    if t == "integer": return int
    if t == "boolean": return bool
    if t == "array": return List[schema_type(schema.get("items", {}))]
    if t == "object": return Dict[str, Any]
    any_of = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(any_of, list):
        types = [schema_type(x) for x in any_of if x.get("type") != "null"]
        if len(types) == 1: return Optional[types[0]]
        if types: return Union[tuple(types)]
    return Any


def create_pydantic_model_from_schema(name: str, schema: Dict[str, Any]):
    fields = {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
    for field_name, info in properties.items():
        annotation = schema_type(info)
        default = info.get("default", ... if field_name in required else None)
        if field_name not in required and default is None: annotation = Optional[annotation]
        fields[field_name] = (annotation, Field(default=default, description=info.get("description", "")))
    return create_model(f"{name}Input", **fields)


def as_list(value): return value if isinstance(value, list) else []
def as_dict(value): return value if isinstance(value, dict) else {}


def parse_iso_date(value):
    try: return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception: return None


def money(value, currency="USD"):
    try: return f"{currency} {float(value):,.2f}"
    except Exception: return "Unavailable"


def clean_attractions(items):
    name_deny = re.compile(r"\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|bus\s*station|railway|parking|signal|flyover|underpass|bypass|overpass|salai|theru|sandhu|mawatha|marg|nagar|colony|layout|township|extension|ward|sector|block|circle|chowk)\b", re.I)
    category_deny = ("administrative", "populated_place", "residential", "postcode", "suburb", "neighbourhood", "neighborhood", "locality", "commercial.building", "office")
    output, seen = [], set()
    for item in as_list(items):
        if not isinstance(item, dict): continue
        name = str(item.get("name", "")).strip()
        if not name: continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen or name_deny.search(name): continue
        categories = [str(x).lower() for x in item.get("categories", [])]
        if any(any(bad in c for bad in category_deny) for c in categories): continue
        allowed = any(c.startswith("tourism.") or "museum" in c or "culture" in c or "place_of_worship" in c or "historic" in c or c.startswith("natural") or "park" in c or "heritage" in c for c in categories)
        if not allowed: continue
        seen.add(key); output.append(item)
    return output


def clean_restaurants(items):
    name_deny = re.compile(r"\b(street|road|lane|mawatha|marg|salai|theru|sandhu|highway|junction|bus\s*stop|station|nagar|colony)\b", re.I)
    output, seen = [], set()
    for item in as_list(items):
        if not isinstance(item, dict): continue
        name = str(item.get("name", "")).strip()
        if not name or name_deny.search(name): continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen: continue
        categories = [str(x).lower() for x in item.get("categories", [])]
        if not any(c.startswith("catering.") for c in categories): continue
        seen.add(key); output.append(item)
    return output


def extract_json(text: str):
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
    if fenced: return json.loads(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start: raise ValueError("Planner did not return JSON")
    return json.loads(text[start:end + 1])


def build_planner_payload(trip, user_request):
    services, request = trip.get("services", {}), trip.get("request", {})
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
    if not isinstance(plan, dict) or not isinstance(plan.get("days"), list): raise ValueError("Invalid planner JSON")
    valid_dates = set()
    cur = start
    while cur <= end:
        valid_dates.add(cur.isoformat()); cur += timedelta(days=1)
    seen = set(); normalized = []
    for day in plan["days"]:
        if not isinstance(day, dict) or day.get("date") not in valid_dates: continue
        ids = []
        for raw in (day.get("attraction_ids") or [])[:2]:
            try: idx = int(raw)
            except (TypeError, ValueError): continue
            if 0 <= idx < len(attractions) and idx not in seen:
                ids.append(idx); seen.add(idx)
        rid = day.get("restaurant_id")
        try: rid = int(rid) if rid is not None else None
        except (TypeError, ValueError): rid = None
        if rid is not None and not (0 <= rid < len(restaurants)): rid = None
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
    days, cur, ai, ri = [], start, 0, 0
    while cur <= end:
        is_first, is_last = cur == start, cur == end
        selected = [] if is_first or is_last else ([ai] if ai < len(attractions) else [])
        if selected: ai += 1
        days.append({"date": cur.isoformat(), "title": "Arrival" if is_first else ("Departure" if is_last else "Sightseeing"), "attraction_ids": selected, "restaurant_id": ri if restaurants else None, "reason": "Safe fallback using only verified live POIs."})
        if restaurants: ri = (ri + 1) % len(restaurants)
        cur += timedelta(days=1)
    return {"days": days}


def render_trip(trip, plan):
    request, services, live = trip.get("request", {}), trip.get("services", {}), trip.get("liveDataSummary", {})
    origin, city, country = request.get("origin", ""), request.get("destinationCity", ""), request.get("destinationCountry", "")
    depart, return_date = str(request.get("departDate", "")), str(request.get("returnDate", ""))
    travelers, nights = request.get("travelers", 1), int(request.get("durationNights") or 0)
    days = int(request.get("calendarDays") or nights + 1)
    flights, hotels = as_list(services.get("flights")), as_list(services.get("hotels"))
    attractions, restaurants = clean_attractions(services.get("attractions")), clean_restaurants(services.get("restaurants"))
    weather, budget = as_dict(services.get("weather")), as_dict(services.get("budget"))

    lines = ["## ✈️ Trip at a glance\n", f"**{origin} → {city}, {country}**  ", f"**{depart} → {return_date} · {travelers} traveler(s) · {nights} night(s) / {days} calendar day(s)**  ", f"Budget level: **{request.get('budgetLevel', 'budget')}**\n"]

    lines += ["## 🛫 Flights\n"]
    if flights:
        lines.append("| Airline | Price | Departure | Arrival | Duration | Stops |\n|---|---:|---|---|---:|---:|")
        for f in flights[:5]:
            stops = int(f.get("stops", 0) or 0)
            lines.append(f"| {f.get('airline', 'Unknown')} | {money(f.get('price'), f.get('currency', 'USD'))} | {f.get('departure', '—')} | {f.get('arrival', '—')} | {f.get('duration', '—')} | {'Non-stop' if stops == 0 else f'{stops} stop(s)'} |")
        lines.append("\n*Live provider results for this search; prices and availability can change.*\n")
    else:
        lines.append(f"**Live flights unavailable.** {as_dict(services.get('flights')).get('error', 'No live flight options were returned.')}\n")

    lines += ["## 🏨 Hotels\n"]
    if hotels:
        lines.append("| Hotel | Nightly | Rating | Reviews |\n|---|---:|---:|---:|")
        for h in hotels[:6]:
            rating = f"{float(h['rating']):.1f}" if isinstance(h.get("rating"), (int, float)) else "—"
            lines.append(f"| {h.get('name', 'Unknown')} | {money(h.get('price'), h.get('currency', 'USD'))} | {rating} | {h.get('reviews', '—')} |")
        lines.append(f"\n*Live hotel rates returned for {depart} → {return_date}.*\n")
    else:
        lines.append(f"**Live hotels unavailable.** {as_dict(services.get('hotels')).get('error', 'No live hotel options were returned.')}\n")

    lines += ["## 📍 Things to do\n"]
    if attractions:
        used = []
        for d in plan.get("days", []): used.extend(d.get("attraction_ids", []))
        for idx in dict.fromkeys(used):
            if 0 <= idx < len(attractions):
                p = attractions[idx]
                lines.append(f"- **{p.get('name')}**" + (f" — {p.get('description')}" if p.get("description") else ""))
        if not used: lines.append("No attractions were selected by the itinerary planner from the verified results.")
    else:
        lines.append("No high-quality live tourist attractions were returned. I won't pad the itinerary with roads, stations, or random map objects.")
    lines.append("")

    lines += ["## 🍽️ Food picks\n"]
    if restaurants:
        for r in restaurants[:8]: lines.append(f"- **{r.get('name')}**" + (f" — {r.get('description')}" if r.get("description") else ""))
        lines.append("\n*Recommendations only; no reservation is implied.*\n")
    else:
        lines.append("No verified restaurant results were returned.\n")

    lines += ["## 🌦️ Weather\n"]
    weather_rows = as_list(weather.get("results"))
    if weather_rows:
        lines.append("| Date | Temp | Feels like | Conditions | Humidity | Rain |\n|---|---:|---:|---|---:|---:|")
        for w in weather_rows: lines.append(f"| {w.get('date', '—')} | {w.get('temperature', '—')}°C | {w.get('feelsLike', '—')}°C | {w.get('description', '—')} | {w.get('humidity', '—')}% | {w.get('precipitationProbability', '—')}% |")
        c = weather.get("coverage", {})
        if c: lines.append(f"\n*Live forecast coverage: **{c.get('returnedStart')} → {c.get('returnedEnd')}**. No weather is extrapolated beyond the provider's returned dates.*\n")
    else:
        lines.append(f"**Live weather unavailable.** {weather.get('error', 'No forecast data was returned for the requested dates.')}\n")

    lines += ["## 💰 Budget\n"]
    if budget:
        cur = budget.get("currency", "USD"); br = budget.get("breakdown", {})
        lines.append(f"**Generic estimate:** {money(budget.get('total_budget'), cur)}")
        lines.append(f"- Flights estimate: {money(br.get('flights_estimate'), cur)}")
        lines.append(f"- Accommodation estimate: {money(br.get('accommodation_estimate'), cur)}")
        lines.append(f"- Daily expenses estimate: {money(br.get('daily_expenses_estimate'), cur)}")
        lines.append("\n*Generic planning estimate only — not a live booking total.*")
    if live.get("complete"):
        lines.append(f"\n**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'), live.get('currency', 'USD'))}")
        lines.append("Includes the cheapest returned live flight offer + cheapest returned hotel nightly rate × nights. It excludes food, local transport, activities, and provider taxes/fees not included in returned prices.")
    else:
        lines.append("\n**Live-data subtotal:** incomplete because a usable live flight or hotel price was missing.")
    lines.append("")

    lines += ["## 🗓️ Suggested itinerary\n"]
    planner_days = plan.get("days", [])
    if not planner_days:
        lines.append("I don't have enough verified attraction data to create a responsible day-by-day itinerary without inventing places.")
    else:
        for i, d in enumerate(planner_days, start=1):
            lines.append(f"### Day {i} · {d.get('date')} · {d.get('title')}")
            if i == 1: lines.append("- ✈️ Arrival / check-in")
            ids = d.get("attraction_ids", [])
            for j, idx in enumerate(ids):
                if 0 <= idx < len(attractions): lines.append(f"- **{'Morning' if j == 0 else 'Afternoon'}:** {attractions[idx].get('name')}")
            rid = d.get("restaurant_id")
            if rid is not None and 0 <= rid < len(restaurants): lines.append(f"- 🍽️ **Food:** {restaurants[rid].get('name')}")
            if d.get("reason"): lines.append(f"- _Why this works:_ {d.get('reason')}")
            if i == len(planner_days): lines.append("- 🧳 Check-out / departure")
            if not ids and rid is None: lines.append("- Keep this period flexible rather than inventing another attraction.")
            lines.append("")

    lines += ["## ⚠️ Notes\n", "- Live prices and availability can change before booking.", "- Generic budget estimates are not live booking totals.", "- Weather is shown only for dates covered by the live forecast provider.", "- The itinerary uses only provider-returned attraction and restaurant IDs."]
    return "\n".join(lines)


async def needs_fresh_tool_data(user_msg: str, trip_data: dict, last_answer: str) -> bool:
    if not trip_data: return True
    prompt = f"""Classify exactly REUSE or FRESH.\nExisting data keys: {', '.join(trip_data.keys())}\nLast answer: {last_answer[:500]}\nNew user message: {user_msg}\nREUSE only if completely answerable from existing trip data. FRESH for a new destination, dates, origin, travelers, category, or live information.\nReply exactly one word."""
    try:
        router = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=5)
        result = await router.ainvoke([HumanMessage(content=prompt)])
        return not (result.content or "").strip().upper().startswith("REUSE")
    except Exception:
        return True


def current_trip_is_valid(trip):
    req = trip.get("request", {})
    start, end, today = parse_iso_date(req.get("departDate")), parse_iso_date(req.get("returnDate")), date.today()
    if not start or not end: return False, "Please provide valid departure and return dates in YYYY-MM-DD format."
    if start < today: return False, f"The departure date {start.isoformat()} is in the past. Today is {today.isoformat()}."
    if end <= start: return False, "The return date must be after the departure date."
    return True, ""


async def build_itinerary_with_llm(llm, trip, user_request):
    req = trip.get("request", {})
    start, end = parse_iso_date(req.get("departDate")), parse_iso_date(req.get("returnDate"))
    attractions = clean_attractions(trip.get("services", {}).get("attractions"))
    restaurants = clean_restaurants(trip.get("services", {}).get("restaurants"))
    if not start or not end: return {"days": []}

    payload = build_planner_payload(trip, user_request)
    prompt = f"""
You are the itinerary planner inside a travel agent. Your ONLY job is to choose and order items from VERIFIED provider data below.
Return JSON ONLY. No Markdown. No prose outside JSON.

Schema:
{{"days":[{{"date":"YYYY-MM-DD","title":"short meaningful title","attraction_ids":[0,1],"restaurant_id":0,"reason":"one short sentence"}}]}}

Rules:
- Use only attraction_ids and restaurant_ids supplied below.
- Never invent a place, restaurant, activity, price, or weather claim.
- Dates must be within the trip dates.
- Use each attraction at most once.
- 0–2 attractions per day. Arrival/departure days should be light.
- Prefer major/historic/cultural/religious/nature sites over weak map POIs.
- Only group places when their returned addresses/descriptions make that grouping defensible.
- If there are not enough strong attractions, leave days intentionally light.
- Restaurant suggestions must use only supplied restaurant IDs.
- The reason must be based only on the supplied data.

USER REQUEST:
{user_request}

VERIFIED TRIP DATA:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
    try:
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        return validate_plan(extract_json(result.content), attractions, restaurants, start, end)
    except Exception:
        return fallback_plan(attractions, restaurants, start, end)


async def run_agent(chat_history, trip_data, chat_container):
    async with AsyncExitStack() as stack:
        status = chat_container.empty(); status.info("🔌 Connecting to MCP server…")
        try:
            transport = await stack.enter_async_context(sse_client(server_url))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            status.info("🧰 Loading travel services…")
            mcp_tools = await session.list_tools()
            tools = []
            for tool in mcp_tools.tools:
                async def call_mcp_tool(tool_name=tool.name, **kwargs):
                    return await session.call_tool(tool_name, arguments=kwargs)
                tools.append(StructuredTool.from_function(func=None, coroutine=call_mcp_tool, name=tool.name, description=tool.description or "Travel MCP tool", args_schema=create_pydantic_model_from_schema(tool.name, tool.input_schema)))

            endpoint = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.3-70B-Instruct", task="text-generation", max_new_tokens=1800, temperature=0.01, huggingfacehub_api_token=os.environ["HF_TOKEN"])
            llm = ChatHuggingFace(llm=endpoint)

            latest_user = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), "")
            last_answer = next((m["content"] for m in reversed(chat_history) if m["role"] == "assistant"), "")
            fresh = await needs_fresh_tool_data(latest_user, trip_data, last_answer)

            if fresh:
                trip_data.clear()
                status.info("🧠 Extracting trip request…")
                bundle_tool = next((t for t in tools if t.name == "build_trip_data"), None)
                if not bundle_tool: raise RuntimeError("MCP server does not expose build_trip_data.")

                extractor = llm.bind_tools([bundle_tool])
                first = await extractor.ainvoke([HumanMessage(content=latest_user)])
                call = next((c for c in (first.tool_calls or []) if c.get("name") == "build_trip_data"), None)
                if not call:
                    status.empty(); return "## 🧭 I need a little more information\n\nPlease provide the origin, destination, departure date, return date, and number of travelers."

                args = dict(call.get("args") or {})
                start, end, today = parse_iso_date(args.get("departDate")), parse_iso_date(args.get("returnDate")), date.today()
                if not start or not end:
                    status.empty(); return "## ⚠️ Invalid dates\n\nPlease provide departure and return dates in YYYY-MM-DD format."
                if start < today:
                    status.empty(); return f"## ⚠️ Past travel date\n\nThe departure date **{start.isoformat()}** is in the past. Today is **{today.isoformat()}**."
                if end <= start:
                    status.empty(); return "## ⚠️ Invalid trip dates\n\nThe return date must be after the departure date."

                with st.expander("🧰 Live services used", expanded=False):
                    st.caption("Flights · Hotels · Attractions · Restaurants · Weather · Budget")
                    st.json(args)

                status.info("⚡ Fetching live trip data…")
                result = await bundle_tool.coroutine(**args)
                trip = json.loads(result.content[0].text)
                trip_data["build_trip_data"] = trip
                valid, error = current_trip_is_valid(trip)
                if not valid:
                    status.empty(); return f"## ⚠️ I can't plan this trip yet\n\n**{error}**"

                status.info("🧠 Planning itinerary with Llama 3.3 70B…")
                plan = await build_itinerary_with_llm(llm, trip, latest_user)
                status.empty()
                return render_trip(trip, plan)

            existing = trip_data.get("build_trip_data")
            if isinstance(existing, dict):
                valid, error = current_trip_is_valid(existing)
                if not valid:
                    status.empty(); return f"## ⚠️ Current trip is no longer valid\n\n**{error}**"
                status.info("♻️ Reusing verified live trip data…")
                plan = await build_itinerary_with_llm(llm, existing, latest_user)
                status.empty()
                return render_trip(existing, plan)

            status.empty(); return "I don't have verified live trip data for this session yet. Please submit a fresh trip request."
        except Exception as exc:
            status.error(f"Something went wrong: {exc}")
            return None


if "messages" not in st.session_state: st.session_state.messages = []
if "trip_data" not in st.session_state: st.session_state.trip_data = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

prompt = st.chat_input("Where are you going? Try: Chennai → Madurai, Aug 20–25, 1 traveler")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        response = asyncio.run(run_agent(st.session_state.messages, st.session_state.trip_data, st.empty()))
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
