import asyncio
import json
import os
from contextlib import AsyncExitStack
from datetime import datetime, timedelta
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
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    for field_name, info in properties.items():
        annotation = schema_type(info)
        default = info.get("default", ... if field_name in required else None)
        if field_name not in required and default is None: annotation = Optional[annotation]
        fields[field_name] = (annotation, Field(default=default, description=info.get("description", "")))
    return create_model(f"{name}Input", **fields)


CLASSIFIER_MODEL = "openai/gpt-oss-20b"


async def needs_fresh_tool_data(user_msg: str, trip_data: dict, last_answer: str) -> bool:
    if not trip_data: return True
    prompt = f"""Classify exactly REUSE or FRESH.
Existing data keys: {', '.join(trip_data.keys())}
Last answer: {last_answer[:500]}
New user message: {user_msg}
REUSE only if the new request is answerable from existing trip data.
FRESH for new destination, dates, origin, travelers, category, or live information.
Reply with exactly one word."""
    try:
        router = ChatGroq(model=CLASSIFIER_MODEL, temperature=0, max_tokens=5)
        result = await router.ainvoke([HumanMessage(content=prompt)])
        return not (result.content or "").strip().upper().startswith("REUSE")
    except Exception:
        return True


def json_obj(value):
    if isinstance(value, dict): return value
    try: return json.loads(value)
    except Exception: return {}


def as_list(value):
    return value if isinstance(value, list) else []


def money(value, currency="USD"):
    try: return f"{currency} {float(value):,.2f}"
    except Exception: return "Price unavailable"


def dates_between(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def clean_attractions(attractions):
    bad = ("road", "street", "highway", "bus stop", "bus station", "parking", "junction", "roundabout", "signal")
    output, seen = [], set()
    for item in as_list(attractions):
        name = str(item.get("name", "")).strip()
        if not name: continue
        key = name.lower().strip()
        if key in seen: continue
        text = f"{name} {item.get('description') or ''} {item.get('kinds') or ''}".lower()
        categories = [str(x).lower() for x in item.get("categories", [])]
        has_tourism = any(x.startswith("tourism.") for x in categories)
        if not has_tourism and any(term in text for term in bad): continue
        if "statue" in text and not any(x in text for x in ("temple", "monument", "museum", "historic", "memorial")): continue
        seen.add(key); output.append(item)
    return output


def render_trip(trip):
    request = trip.get("request", {})
    services = trip.get("services", {})
    live = trip.get("liveDataSummary", {})

    origin = request.get("origin", "")
    city = request.get("destinationCity", "")
    country = request.get("destinationCountry", "")
    depart = request.get("departDate", "")
    return_date = request.get("returnDate", "")
    travelers = request.get("travelers", 1)
    nights = int(request.get("durationNights", 0) or 0)
    calendar_days = int(request.get("calendarDays", nights + 1) or nights + 1)

    flights = as_list(services.get("flights"))
    hotels = as_list(services.get("hotels"))
    attractions = clean_attractions(services.get("attractions"))
    restaurants = as_list(services.get("restaurants"))
    weather_obj = json_obj(services.get("weather"))
    budget = json_obj(services.get("budget"))

    lines = []
    lines.append("## ✈️ Trip at a glance\n")
    lines.append(f"**{origin} → {city}, {country}**  ")
    lines.append(f"**{depart} → {return_date} · {travelers} traveler(s) · {nights} night(s) / {calendar_days} calendar day(s)**  ")
    lines.append(f"Budget level: **{request.get('budgetLevel', 'mid-range')}**\n")

    lines.append("## 🛫 Flights\n")
    if flights:
        lines.append("| Airline | Price | Departure | Arrival | Duration | Stops |\n|---|---:|---|---|---:|---:|")
        for flight in flights[:5]:
            stops = int(flight.get('stops', 0) or 0)
            lines.append(f"| {flight.get('airline', 'Unknown')} | {money(flight.get('price'), flight.get('currency', 'USD'))} | {flight.get('departure', '—')} | {flight.get('arrival', '—')} | {flight.get('duration', '—')} | {'Non-stop' if stops == 0 else str(stops) + ' stop(s)'} |")
        lines.append("\n*Live provider prices for this search. Availability may change.*\n")
    else:
        error = json_obj(services.get("flights")).get("error") or services.get("flights")
        lines.append(f"**Live flights unavailable.** {error or 'No flight options were returned.'}\n")

    lines.append("## 🏨 Hotels\n")
    if hotels:
        lines.append("| Hotel | Nightly | Rating | Reviews |\n|---|---:|---:|---:|")
        for hotel in hotels[:6]:
            rating = hotel.get("rating")
            rating_text = f"{float(rating):.1f}" if isinstance(rating, (int, float)) else "—"
            lines.append(f"| {hotel.get('name', 'Unknown')} | {money(hotel.get('price'), hotel.get('currency', 'USD'))} | {rating_text} | {hotel.get('reviews', '—')} |")
        lines.append(f"\n*Rates returned for {depart} → {return_date}.*\n")
    else:
        err = services.get("hotels")
        err_text = err.get("error") if isinstance(err, dict) else str(err or "No hotel options were returned.")
        lines.append(f"**Live hotels unavailable.** {err_text}\n")

    lines.append("## 📍 Things to do\n")
    if attractions:
        lines.append("Only POIs that survived the attraction-quality filter are shown.\n")
        for place in attractions[:8]:
            name = place.get("name", "Unknown place")
            desc = place.get("description")
            lines.append(f"- **{name}**" + (f" — {desc}" if desc else ""))
        lines.append("")
    else:
        lines.append("**No high-quality live tourist attractions were returned.** I won't pad this with roads, bus stops, or random map infrastructure.\n")

    lines.append("## 🍽️ Food picks\n")
    if restaurants:
        for restaurant in restaurants[:8]:
            name = restaurant.get("name", "Unknown restaurant")
            desc = restaurant.get("description") or ""
            lines.append(f"- **{name}**" + (f" — {desc}" if desc else ""))
        lines.append("\n*Recommendations only; no reservation is implied.*\n")
    else:
        lines.append("**No live restaurant results were returned.**\n")

    lines.append("## 🌦️ Weather\n")
    weather_results = weather_obj.get("results", []) if isinstance(weather_obj, dict) else []
    if weather_results:
        lines.append("| Date | Temp | Feels like | Conditions | Humidity | Rain |\n|---|---:|---:|---|---:|---:|")
        for item in weather_results:
            lines.append(f"| {item.get('date', '—')} | {item.get('temperature', '—')}°C | {item.get('feelsLike', '—')}°C | {item.get('description', '—')} | {item.get('humidity', '—')}% | {item.get('precipitationProbability', '—')}% |")
        coverage = weather_obj.get("coverage", {})
        returned_start = coverage.get("returnedStart")
        returned_end = coverage.get("returnedEnd")
        if returned_start and returned_end:
            covered = returned_start if returned_start == returned_end else f"{returned_start} → {returned_end}"
            lines.append(f"\n*Live forecast coverage currently returned: **{covered}**. Later dates are not extrapolated.*\n")
    else:
        err = weather_obj.get("error") if isinstance(weather_obj, dict) else services.get("weather")
        lines.append(f"**Live weather unavailable for the requested dates.** {err or ''}\n")

    lines.append("## 💰 Budget\n")
    if budget:
        bcurrency = budget.get("currency", "USD")
        total = budget.get("total_budget")
        breakdown = budget.get("breakdown", {})
        lines.append(f"**Generic estimate:** {money(total, bcurrency)}")
        lines.append(f"- Flight estimate: {money(breakdown.get('flights_estimate'), bcurrency)}")
        lines.append(f"- Accommodation estimate: {money(breakdown.get('accommodation_estimate'), bcurrency)}")
        lines.append(f"- Daily expenses estimate: {money(breakdown.get('daily_expenses_estimate'), bcurrency)}")
        lines.append("\n*This is a planning estimate, not a live quote.*\n")
    if live.get("complete"):
        lines.append(f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'), live.get('currency', 'USD'))}")
        lines.append("Includes the cheapest returned live flight offer + cheapest returned hotel rate × nights. It excludes food, local transport, activities, and taxes/fees not included by providers.\n")
    else:
        lines.append("**Live-data subtotal:** incomplete because a live flight or hotel price was missing.\n")

    lines.append("## 🗓️ Suggested itinerary\n")
    itinerary_dates = dates_between(depart, return_date) if depart and return_date else []
    useful_attractions = attractions[:max(1, min(6, len(attractions)))]
    useful_restaurants = restaurants[:max(1, min(6, len(restaurants)))]

    for idx, day in enumerate(itinerary_dates):
        lines.append(f"### Day {idx + 1} · {day.strftime('%a, %d %b %Y')}")
        if idx == 0:
            lines.append("- **Arrival:** Travel from the origin and check in after arrival.")
            if useful_attractions:
                lines.append(f"- **Evening:** Easy first stop at **{useful_attractions[0].get('name')}** if time/energy allows.")
        elif idx == len(itinerary_dates) - 1:
            lines.append("- **Morning:** Check out and keep the final morning flexible.")
            lines.append("- **Departure:** Return journey after check-out.")
        else:
            if useful_attractions:
                a1 = useful_attractions[(idx - 1) % len(useful_attractions)]
                lines.append(f"- **Morning:** **{a1.get('name')}**")
                if idx % 2 == 1 and len(useful_attractions) > 1:
                    a2 = useful_attractions[idx % len(useful_attractions)]
                    if a2.get('name') != a1.get('name'):
                        lines.append(f"- **Afternoon:** **{a2.get('name')}**")
            else:
                lines.append("- **Day plan:** Keep the day flexible; the live place feed did not return enough quality attractions.")
            if useful_restaurants:
                r = useful_restaurants[(idx - 1) % len(useful_restaurants)]
                lines.append(f"- **Food:** **{r.get('name')}**")
        lines.append("")

    lines.append("## ⚠️ Notes\n")
    lines.append("- Live prices and availability can change before booking.")
    lines.append("- The itinerary uses live POIs that passed the service's quality filter; it does not invent landmarks.")
    if weather_results:
        lines.append("- Weather is shown only for dates actually covered by the live forecast provider.")
    return "\n".join(lines)


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
            llm_tools = llm.bind_tools(tools)

            latest_user = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), "")
            last_answer = next((m["content"] for m in reversed(chat_history) if m["role"] == "assistant"), "")
            fresh = await needs_fresh_tool_data(latest_user, trip_data, last_answer)
            if fresh and trip_data: trip_data.clear()

            if fresh:
                status.info("🧠 Planning with live travel services…")
                first = await llm_tools.ainvoke([
                    SystemMessage(content="For a complete first trip, call build_trip_data when available. Do not fabricate missing dates or origins."),
                    HumanMessage(content=latest_user)
                ])
                bundle_tool = next((t for t in tools if t.name == "build_trip_data"), None)
                bundle_call = next((c for c in (first.tool_calls or []) if c.get("name") == "build_trip_data"), None)

                if bundle_tool and bundle_call:
                    with st.expander("🧰 Live services used", expanded=False):
                        st.caption("Flights · Hotels · Attractions · Restaurants · Weather · Budget")
                    status.info("⚡ Fetching live trip data…")
                    result = await bundle_tool.coroutine(**bundle_call["args"])
                    content = result.content[0].text
                    trip = json_obj(content)
                    trip_data["build_trip_data"] = trip
                    status.empty()
                    return render_trip(trip)

            # Follow-ups or fallback: use structured trip data, not invented values.
            messages = [SystemMessage(content="Answer using only CURRENT TRIP DATA. Never invent prices, places, or dates.")]
            messages.append(SystemMessage(content="CURRENT TRIP DATA:\n" + json.dumps(trip_data, ensure_ascii=False)[:14000]))
            for m in [x for x in chat_history if x["role"] in ("user", "assistant")][-8:]:
                messages.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
            status.info("♻️ Reusing the existing trip data…")
            final = await llm.ainvoke(messages)
            status.empty()
            return final.content or "I couldn't generate a response from the available live data."
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
