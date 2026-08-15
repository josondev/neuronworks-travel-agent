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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

nest_asyncio.apply()
st.set_page_config(page_title="Neuronworks Travel Agent", page_icon="✈️", layout="wide")

st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 10% 0%, #1d4ed8 0, transparent 35%), radial-gradient(circle at 90% 10%, #7c3aed 0, transparent 32%), #080b14; }
.block-container { max-width: 1180px; padding-top: 1.7rem; padding-bottom: 6rem; }
.hero { padding: 28px 30px; border-radius: 24px; margin-bottom: 20px; background: linear-gradient(135deg, rgba(37,99,235,.32), rgba(124,58,237,.27)); border: 1px solid rgba(255,255,255,.12); box-shadow: 0 18px 60px rgba(0,0,0,.25); }
.hero h1 { margin: 0; color: #fff; font-size: 2.2rem; letter-spacing: -.04em; }
.hero p { margin: 8px 0 0; color: #cbd5e1; font-size: .98rem; }
.badge { display:inline-block; padding:5px 11px; border-radius:999px; background:rgba(255,255,255,.10); color:#e2e8f0; font-size:.76rem; border:1px solid rgba(255,255,255,.12); }
section[data-testid="stSidebar"] { background: rgba(8,11,20,.96); border-right: 1px solid rgba(255,255,255,.08); }
div[data-testid="stChatMessage"] { border: 1px solid rgba(255,255,255,.09); border-radius: 20px; padding: 1.15rem 1.25rem; margin: .8rem 0; background: rgba(15,23,42,.80); box-shadow: 0 10px 30px rgba(0,0,0,.18); }
div[data-testid="stChatMessageContent"] { color:#e5e7eb; }
div[data-testid="stChatMessageContent"] h1, div[data-testid="stChatMessageContent"] h2, div[data-testid="stChatMessageContent"] h3 { color:#fff; }
div[data-testid="stChatMessageContent"] table { border-radius:12px; overflow:hidden; }
div[data-testid="stExpander"] { border:1px solid rgba(255,255,255,.09); border-radius:14px; }
.muted { color:#94a3b8; font-size:.82rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="badge">● LIVE MCP TRAVEL INTELLIGENCE</div>
  <h1>✈️ Neuronworks Travel Agent</h1>
  <p>Live flights · hotels · places · restaurants · weather · currency · budget</p>
</div>
""", unsafe_allow_html=True)

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
    if not isinstance(schema, dict):
        return Any
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        try:
            return Literal[tuple(schema["enum"])]
        except TypeError:
            pass
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
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    for field_name, info in properties.items():
        annotation = schema_type(info)
        default = info.get("default", ... if field_name in required else None)
        if field_name not in required and default is None:
            annotation = Optional[annotation]
        fields[field_name] = (annotation, Field(default=default, description=info.get("description", "")))
    return create_model(f"{name}Input", **fields)


def as_list(value):
    return value if isinstance(value, list) else []


def obj(value):
    return value if isinstance(value, dict) else {}


def money(value, currency="USD"):
    try:
        return f"{currency} {float(value):,.2f}"
    except Exception:
        return "Unavailable"


def parse_iso_date(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None


def clean_attractions(items):
    banned = re.compile(r"(road|street|highway|lane|path|junction|roundabout|bus stop|bus station|railway|parking|signal)$", re.I)
    out, seen = [], set()
    for item in as_list(items):
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen or banned.search(name):
            continue
        categories = [str(x).lower() for x in item.get("categories", [])]
        if not any(
            c.startswith("tourism.") or
            "museum" in c or
            "culture" in c or
            "place_of_worship" in c or
            "historic" in c or
            c.startswith("natural") or
            "park" in c
            for c in categories
        ):
            continue
        seen.add(key)
        out.append(item)
    return out


def clean_restaurants(items):
    bad_name = re.compile(r"(street|road|lane|mawatha|marg|highway|junction|bus stop|station)$", re.I)
    out, seen = [], set()
    for item in as_list(items):
        name = str(item.get("name", "")).strip()
        if not name or bad_name.search(name):
            continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen:
            continue
        categories = [str(x).lower() for x in item.get("categories", [])]
        if not any(c.startswith("catering.") for c in categories):
            continue
        seen.add(key)
        out.append(item)
    return out


def render_trip(trip):
    if trip.get("planningBlocked"):
        return (
            "## ⚠️ I can't plan this trip yet\n\n"
            f"**{trip.get('error', 'The requested dates are not valid for live travel search.')}**\n\n"
            "Please give me future departure and return dates."
        )

    request = trip.get("request", {})
    services = trip.get("services", {})
    live = trip.get("liveDataSummary", {})

    depart = str(request.get("departDate", ""))
    return_date = str(request.get("returnDate", ""))
    today = date.today()
    start = parse_iso_date(depart)
    end = parse_iso_date(return_date)

    if start and start < today:
        return f"## ⚠️ Past travel date\n\nThe departure date **{depart}** has already passed. Today is **{today.isoformat()}**. Please choose a future trip."
    if end and end <= today and start:
        return f"## ⚠️ Past travel date\n\nThe return date **{return_date}** has already passed. Today is **{today.isoformat()}**. Please choose a future trip."

    origin = request.get("origin", "")
    city = request.get("destinationCity", "")
    country = request.get("destinationCountry", "")
    travelers = request.get("travelers", 1)
    nights = int(request.get("durationNights") or 0)
    days = int(request.get("calendarDays") or nights + 1)

    flights = as_list(services.get("flights"))
    hotels = as_list(services.get("hotels"))
    attractions = clean_attractions(services.get("attractions"))
    restaurants = clean_restaurants(services.get("restaurants"))
    weather = obj(services.get("weather"))
    budget = obj(services.get("budget"))

    # --- FIX: the itinerary loop below needs up to `days * 2` attractions
    # (2 per day). Previously "Things to do" was hardcoded to attractions[:8]
    # while the itinerary indexed into the full `attractions` list, so on
    # trips longer than 4 days the itinerary silently pulled in places that
    # were never shown to the user in "Things to do" (e.g. index 8, 9...).
    # We now size a single shared pool to what the itinerary actually needs,
    # and both sections read from that same pool so nothing appears in the
    # itinerary without having been listed above first.
    needed_for_itinerary = max(days, 1) * 2
    itinerary_pool = attractions[:needed_for_itinerary]

    lines = []
    lines.append("## ✈️ Trip at a glance\n")
    lines.append(f"**{origin} → {city}, {country}**  ")
    lines.append(f"**{depart} → {return_date} · {travelers} traveler(s) · {nights} night(s) / {days} calendar day(s)**\n")

    lines.append("## 🛫 Flights\n")
    if flights:
        lines.append("| Airline | Price | Departure | Arrival | Duration | Stops |\n|---|---:|---|---|---:|---:|")
        for f in flights[:5]:
            stops = int(f.get("stops", 0) or 0)
            lines.append(f"| {f.get('airline', 'Unknown')} | {money(f.get('price'), f.get('currency', 'USD'))} | {f.get('departure', '—')} | {f.get('arrival', '—')} | {f.get('duration', '—')} | {'Non-stop' if stops == 0 else f'{stops} stop(s)'} |")
        lines.append("")
    else:
        err = obj(services.get("flights")).get("error") or "No live flight options were returned."
        lines.append(f"**Live flights unavailable.** {err}\n")

    lines.append("## 🏨 Hotels\n")
    if hotels:
        lines.append("| Hotel | Nightly | Rating | Reviews |\n|---|---:|---:|---:|")
        for h in hotels[:6]:
            rating = f"{float(h['rating']):.1f}" if isinstance(h.get("rating"), (int, float)) else "—"
            lines.append(f"| {h.get('name', 'Unknown')} | {money(h.get('price'), h.get('currency', 'USD'))} | {rating} | {h.get('reviews', '—')} |")
        lines.append("")
    else:
        err = obj(services.get("hotels")).get("error") or "No live hotel options were returned."
        lines.append(f"**Live hotels unavailable.** {err}\n")

    lines.append("## 📍 Things to do\n")
    if itinerary_pool:
        for p in itinerary_pool[:8]:  # cap the summary bullet list to 8 for readability
            desc = p.get("description")
            lines.append(f"- **{p.get('name', 'Unknown')}**" + (f" — {desc}" if desc else ""))
        if len(itinerary_pool) > 8:
            lines.append(f"- …and {len(itinerary_pool) - 8} more used directly in the itinerary below")
    else:
        lines.append("No strong tourist attractions were returned by the live places service. I won't pad the itinerary with roads, stations, or random map objects.")
    lines.append("")

    lines.append("## 🍽️ Food picks\n")
    if restaurants:
        for r in restaurants[:8]:
            lines.append(f"- **{r.get('name', 'Unknown restaurant')}**")
        lines.append("\n*Recommendations only; no reservation is implied.*")
    else:
        lines.append("No verified restaurant results were returned.")
    lines.append("")

    lines.append("## 🌦️ Weather\n")
    weather_rows = as_list(weather.get("results"))
    if weather_rows:
        lines.append("| Date | Temp | Feels like | Conditions | Humidity | Rain |\n|---|---:|---:|---|---:|---:|")
        for w in weather_rows:
            lines.append(f"| {w.get('date', '—')} | {w.get('temperature', '—')}°C | {w.get('feelsLike', '—')}°C | {w.get('description', '—')} | {w.get('humidity', '—')}% | {w.get('precipitationProbability', '—')}% |")
        lines.append("")
        coverage = weather.get("coverage", {})
        if coverage:
            lines.append(f"*Live forecast coverage returned: **{coverage.get('returnedStart')} → {coverage.get('returnedEnd')}**. Later dates are not extrapolated.*")
    else:
        lines.append(f"**Live weather unavailable.** {weather.get('error', 'No forecast data was returned for the requested dates.')}")
    lines.append("")

    lines.append("## 💰 Budget\n")
    if budget:
        currency = budget.get("currency", "USD")
        lines.append(f"**Generic estimate:** {money(budget.get('total_budget'), currency)}")
        br = budget.get("breakdown", {})
        lines.append(f"- Flights estimate: {money(br.get('flights_estimate'), currency)}")
        lines.append(f"- Accommodation estimate: {money(br.get('accommodation_estimate'), currency)}")
        lines.append(f"- Daily expenses estimate: {money(br.get('daily_expenses_estimate'), currency)}")
        lines.append("*Generic planning estimate only — not a live quote.*")
    if live.get("complete"):
        lines.append(f"\n**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'), live.get('currency', 'USD'))}")
        lines.append("This subtotal is flight + cheapest hotel rate × nights. It excludes food, local transport, activities and provider fees/taxes not included in the returned prices.")
    else:
        lines.append("\n**Live-data subtotal:** incomplete because a usable live flight or hotel price was missing.")
    lines.append("")

    lines.append("## 🗓️ Suggested itinerary\n")
    if not itinerary_pool:
        lines.append("I won't invent an itinerary without trustworthy attraction data. Use the verified places above once the places service returns suitable tourist POIs.")
    else:
        # Each attraction appears at most once. We use up to one primary + one secondary per day.
        # NOTE: now reads from `itinerary_pool`, the same list shown under "Things to do" above,
        # so every place named here was already surfaced to the user.
        schedule = []
        for d in range(max(days, 1)):
            current = start + timedelta(days=d) if start else None
            if current and current > end:
                break
            p1 = itinerary_pool[d * 2] if d * 2 < len(itinerary_pool) else None
            p2 = itinerary_pool[d * 2 + 1] if d * 2 + 1 < len(itinerary_pool) else None
            restaurant = restaurants[d % len(restaurants)] if restaurants else None
            schedule.append((current, p1, p2, restaurant))

        for idx, (current, p1, p2, restaurant) in enumerate(schedule, start=1):
            title = current.strftime("%a, %d %b %Y") if current else f"Day {idx}"
            lines.append(f"### Day {idx} · {title}")
            if idx == 1:
                lines.append("- Arrival / hotel check-in")
            if p1:
                lines.append(f"- Visit **{p1.get('name')}**")
            if p2:
                lines.append(f"- Then visit **{p2.get('name')}**")
            if restaurant:
                lines.append(f"- Food suggestion: **{restaurant.get('name')}**")
            if not p1 and not p2:
                lines.append("- Keep the day flexible rather than inventing additional sights.")
            lines.append("")

    lines.append("## ⚠️ Notes\n")
    lines.append("- Live prices and availability can change before booking.")
    lines.append("- Generic budget estimates are not live booking totals.")
    lines.append("- Weather is shown only for dates covered by the live forecast provider.")
    return "\n".join(lines)


async def needs_fresh_tool_data(user_msg: str, trip_data: dict, last_answer: str) -> bool:
    if not trip_data:
        return True
    prompt = f"""Classify exactly REUSE or FRESH.
Existing data keys: {', '.join(trip_data.keys())}
Last answer: {last_answer[:500]}
New user message: {user_msg}
REUSE only if the request is answerable from existing trip data.
FRESH for a new destination, new dates, new origin, new travelers, or new live information.
Reply exactly one word."""
    try:
        router = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=5)
        result = await router.ainvoke([HumanMessage(content=prompt)])
        return not (result.content or "").strip().upper().startswith("REUSE")
    except Exception:
        return True


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
                tools.append(StructuredTool.from_function(
                    func=None,
                    coroutine=call_mcp_tool,
                    name=tool.name,
                    description=tool.description or "Travel MCP tool",
                    args_schema=create_pydantic_model_from_schema(tool.name, tool.input_schema)
                ))

            endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                task="text-generation",
                max_new_tokens=1600,
                temperature=0.01,
                huggingfacehub_api_token=os.environ["HF_TOKEN"]
            )
            llm = ChatHuggingFace(llm=endpoint)
            llm_tools = llm.bind_tools(tools)

            latest_user = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), "")
            last_answer = next((m["content"] for m in reversed(chat_history) if m["role"] == "assistant"), "")
            fresh = await needs_fresh_tool_data(latest_user, trip_data, last_answer)

            if fresh:
                trip_data.clear()
                status.info("🧠 Planning with live travel services…")
                first = await llm_tools.ainvoke([
                    SystemMessage(content=(
                        "For a complete first-trip request, call build_trip_data. "
                        "Do not invent missing dates or origin. If the user's dates are in the past, do not call live providers."
                    )),
                    HumanMessage(content=latest_user)
                ])

                bundle_tool = next((t for t in tools if t.name == "build_trip_data"), None)
                bundle_call = next((c for c in (first.tool_calls or []) if c.get("name") == "build_trip_data"), None)

                if bundle_tool and bundle_call:
                    with st.expander("🧰 Live services used", expanded=False):
                        st.caption("Flights · Hotels · Attractions · Restaurants · Weather · Budget")
                        st.json(bundle_call["args"])
                    result = await bundle_tool.coroutine(**bundle_call["args"])
                    content = result.content[0].text
                    trip = json.loads(content)
                    trip_data["build_trip_data"] = trip
                    status.empty()
                    return render_trip(trip)

                # Fallback when the model uses individual tools instead of the bundle.
                messages = [SystemMessage(content="Use only returned live tool data. Do not invent.")]
                messages.append(HumanMessage(content=latest_user))
                for call in first.tool_calls or []:
                    selected = next((t for t in tools if t.name == call["name"]), None)
                    if not selected:
                        continue
                    with st.expander(f"🧰 {call['name']}", expanded=False):
                        st.json(call["args"])
                    try:
                        result = await selected.coroutine(**call["args"])
                        tool_content = result.content[0].text
                    except Exception as exc:
                        tool_content = json.dumps({"error": str(exc)})
                    messages.append(ToolMessage(tool_call_id=call["id"], name=call["name"], content=tool_content))
                    try:
                        trip_data[call["name"]] = json.loads(tool_content)
                    except Exception:
                        trip_data[call["name"]] = tool_content
                status.info("✨ Writing response…")
                final = await llm.ainvoke(messages)
                status.empty()
                return final.content or "I couldn't generate a response from the available live data."

            # Follow-up: use stored structured data.
            if "build_trip_data" in trip_data:
                status.info("♻️ Reusing your live trip data…")
                answer = render_trip(trip_data["build_trip_data"])
                status.empty()
                return answer

            status.info("♻️ Reusing your existing live trip data…")
            messages = [
                SystemMessage(content="Answer only from CURRENT TRIP DATA. Never invent values."),
                SystemMessage(content="CURRENT TRIP DATA:\n" + json.dumps(trip_data, ensure_ascii=False)[:14000]),
                HumanMessage(content=latest_user)
            ]
            final = await llm.ainvoke(messages)
            status.empty()
            return final.content or "I couldn't generate a response from the available live data."

        except Exception as exc:
            status.error(f"Something went wrong: {exc}")
            return None


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
        response = asyncio.run(run_agent(st.session_state.messages, st.session_state.trip_data, st.empty()))
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
