import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from datetime import date, datetime, timedelta
from typing import Any, Literal, Optional

import nest_asyncio
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from mcp import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel, Field

nest_asyncio.apply()
st.set_page_config(page_title="Neuronworks Travel Agent", page_icon="✈️", layout="wide")

st.markdown("""
<style>
:root{--bg:#080b14;--border:rgba(255,255,255,.10);--muted:#94a3b8}
.stApp{background:radial-gradient(circle at 10% 0%,rgba(37,99,235,.42),transparent 34%),radial-gradient(circle at 90% 10%,rgba(124,58,237,.36),transparent 32%),var(--bg)}
.block-container{max-width:1180px;padding-top:5rem;padding-bottom:6rem}
.hero{padding:26px 28px;border-radius:22px;margin-bottom:18px;background:linear-gradient(135deg,rgba(37,99,235,.28),rgba(124,58,237,.22));border:1px solid var(--border)}
.hero h1{margin:0;color:#fff;font-size:2.2rem}.hero p{margin:8px 0;color:#cbd5e1}
.pill{display:inline-block;padding:5px 11px;border-radius:999px;background:rgba(255,255,255,.09);color:#e2e8f0;font-size:.75rem;border:1px solid var(--border)}
div[data-testid="stChatMessage"]{border:1px solid var(--border);border-radius:18px;padding:1rem 1.1rem;margin:.7rem 0;background:rgba(15,23,42,.82)}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<span class="pill">● LIVE MCP · SEMANTIC ROUTING</span>
<h1>✈️ Neuronworks Travel Agent</h1>
<p>Flights · Hotels · Places · Restaurants · Weather · Budget · Currency</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    server_url = st.text_input("MCP Server URL", "https://neuronworks-travel-agent.onrender.com/sse")
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    if not groq_api_key:
        st.warning("Enter GROQ_API_KEY.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key
    st.success("🟢 Ready")
    st.caption("GPT-OSS 20B · semantic router only")
    st.caption("Fresh travel facts always come from MCP")
    st.caption("City → IATA resolution happens inside MCP before SerpApi")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_trip" not in st.session_state:
    st.session_state.active_trip = None
if "trip_collection" not in st.session_state:
    st.session_state.trip_collection = {}


class Destination(BaseModel):
    city: str = Field(description="Destination city name only. Do not output an airport code.")
    country: Optional[str] = Field(default=None, description="Destination country when confidently known.")


class RouterDecision(BaseModel):
    action: Literal["FETCH", "REUSE", "ASK"]
    missing: Optional[str] = Field(default=None)
    question: Optional[str] = Field(default=None)
    origin: Optional[str] = Field(default=None, description="Origin city name only, never an IATA code.")
    destinations: list[Destination] = Field(default_factory=list)
    departDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    returnDate: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    passengers: Optional[int] = Field(default=None, ge=1)
    budgetLevel: Optional[Literal["budget", "mid-range", "luxury"]] = None
    replace_active: bool = False
    compare_with_active: bool = False


def model(max_tokens=500):
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=max_tokens,
    )


def iso(value: Any):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None


def money(value: Any, currency="USD"):
    try:
        return f"{currency} {float(value):,.2f}"
    except Exception:
        return "Unavailable"


def normalize_city(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).title()


def format_conversation():
    recent = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")][-10:]
    parts = []
    for item in recent:
        text = item["content"]
        if item["role"] == "assistant":
            text = text[:700]
        parts.append(("User" if item["role"] == "user" else "Assistant") + ": " + text)
    return "\n".join(parts) if parts else "(no prior conversation)"


def active_trip_summary():
    context = st.session_state.active_trip
    if not context:
        return "None yet."
    req = context.get("request", {})
    return json.dumps({
        "origin": req.get("origin"),
        "destinationCity": req.get("destinationCity"),
        "destinationCountry": req.get("destinationCountry"),
        "departDate": req.get("departDate"),
        "returnDate": req.get("returnDate"),
        "passengers": req.get("travelers") or req.get("passengers"),
        "budgetLevel": req.get("budgetLevel"),
    }, ensure_ascii=False)


def collection_summary():
    if not st.session_state.trip_collection:
        return "None yet."
    return json.dumps({
        city: {
            "origin": trip.get("request", {}).get("origin"),
            "departDate": trip.get("request", {}).get("departDate"),
            "returnDate": trip.get("request", {}).get("returnDate"),
            "budgetLevel": trip.get("request", {}).get("budgetLevel"),
        }
        for city, trip in st.session_state.trip_collection.items()
    }, ensure_ascii=False)


SYSTEM_PROMPT = f"""
You are an expert, factual AI Travel Agent and the semantic routing brain of a live MCP travel agent. Your goal is to plan realistic, bookable trips using ONLY real-time data returned by MCP.

### CURRENT CONTEXT
- Today's Date: {date.today().isoformat()}
- Resolve relative dates such as next Friday or in 2 days relative to today's date.

### ZERO-HALLUCINATION PROTOCOL
- MCP is the ONLY source of fresh travel facts: flights, hotels, places, restaurants, weather, budget and currency.
- Never invent prices, availability, hotels, flights, attractions or weather.
- Preserve exact prices returned by MCP.
- Generic budget output from MCP is a planning estimate, not a live booking total.
- Airport/city resolution is the MCP server's responsibility. The router must output PLAIN CITY NAMES. Never output or invent IATA codes.

### FLIGHT RULE
- The MCP flight service resolves origin/destination cities to practical airport IATA codes server-side and passes those codes to SerpApi.
- The frontend/router must never perform airport-code lookup.
- IATA codes are used only at the MCP/FlightService → SerpApi boundary.

### FOLLOW-UP / CONTEXT RULES
- Use the full conversation and stored trip context.
- A NEW destination requires fresh MCP data.
- "Do the same", "same trip", "repeat this", "compare this with", and similar wording must be interpreted semantically from the conversation.
- Multiple new destinations must all be returned in `destinations`; the app will fetch them independently and in parallel.
- REUSE is allowed only when the answer can be derived entirely from already-fetched MCP data.
- A comparison must fetch the new candidate with MCP and must not overwrite the active trip.

### COMPARISON / BEST VALUE
- compare_with_active=true when a new destination should be compared with the active trip.
- replace_active=true for the first/full plan or an explicit destination switch.
- "Best value" means lowest returned live flight + hotel subtotal unless the user explicitly asks for another criterion.

### BUDGET RULES
- budgetLevel must be exactly budget, mid-range or luxury.
- calculate_trip_budget is generic planning data only.
- A live subtotal must be computed separately from returned flight + hotel prices.

### REQUIRED ROUTER ACTIONS
- FETCH: obtain fresh live MCP data.
- REUSE: answer from stored MCP data.
- ASK: information truly cannot be recovered from the conversation or stored trips.

### REQUIRED OUTPUT
Return ONLY one valid JSON object. Do NOT call tools. Do NOT use markdown or code fences. Do NOT answer the travel request itself.

Exact JSON shape:
{{
  "action": "FETCH | REUSE | ASK",
  "missing": null,
  "question": null,
  "origin": null,
  "destinations": [{{"city": "", "country": null}}],
  "departDate": null,
  "returnDate": null,
  "passengers": null,
  "budgetLevel": null,
  "replace_active": false,
  "compare_with_active": false
}}

Examples:
1. Complete first request for Chennai → Madurai, Aug 20–25, 1 traveler, budget:
{{"action":"FETCH","missing":null,"question":null,"origin":"Chennai","destinations":[{{"city":"Madurai","country":"India"}}],"departDate":"2026-08-20","returnDate":"2026-08-25","passengers":1,"budgetLevel":"budget","replace_active":true,"compare_with_active":false}}
2. "compare the same for Coimbatore and tell me which is better": FETCH, destinations=[Coimbatore], compare_with_active=true, replace_active=false, inheriting the active trip's origin/dates/travelers/budget.
3. "do the same for Madurai, Kodaikanal and Ooty": FETCH with all three destinations, inheriting origin/dates/travelers/budget.
4. "which hotel is cheapest in Madurai?" when Madurai exists in stored trips: REUSE with question set to the user's question.

### ACTIVE TRIP
{active_trip_summary()}

### STORED TRIPS
{collection_summary()}

### RECENT CONVERSATION
{format_conversation()}
"""


def extract_router_json(raw):
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    for block in re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S):
        try:
            value = json.loads(block.strip())
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None


async def route_request():
    last_error = None
    prompt = SYSTEM_PROMPT
    for attempt in range(2):
        try:
            result = await model(max_tokens=600).ainvoke([HumanMessage(content=prompt)])
            parsed = extract_router_json(result.content)
            if isinstance(parsed, dict) and parsed.get("action") in {"FETCH", "REUSE", "ASK"}:
                return RouterDecision(**parsed)
            last_error = ValueError("Router returned invalid or incomplete JSON")
        except Exception as exc:
            last_error = exc
        prompt = SYSTEM_PROMPT + "\n\nRETRY: Output ONLY the JSON object matching the exact schema. No tool calls. No prose."
    raise RuntimeError(f"Semantic router failed twice: {type(last_error).__name__}: {last_error}")


def validate_request(request):
    start, end = iso(request.get("departDate")), iso(request.get("returnDate"))
    if not request.get("origin") or not request.get("destinationCity"):
        return False, "Please provide a valid origin and destination."
    if not start or not end:
        return False, "Please provide departure and return dates in YYYY-MM-DD format."
    if start < date.today():
        return False, f"The departure date {start.isoformat()} is in the past. Today is {date.today().isoformat()}."
    if end <= start:
        return False, "Return date must be after the departure date."
    return True, ""


async def mcp_trip(session, request):
    result = await asyncio.wait_for(session.call_tool("build_trip_data", arguments=request), timeout=20)
    if not result.content:
        raise RuntimeError("MCP returned no content")
    payload = json.loads(result.content[0].text)
    if payload.get("planningBlocked"):
        raise RuntimeError(payload.get("error", "MCP blocked trip planning"))
    return payload


def listv(services, key):
    value = services.get(key)
    return value if isinstance(value, list) else []


def clean_attractions(items):
    deny = re.compile(r"\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|bus\s*station|railway|parking|signal|flyover|underpass|bypass|overpass|salai|theru|nagar|colony|layout|township|extension|ward|sector|block|circle|chowk|hospital|water\s*works|car\s*shelter)\b|நகரம்|சாலை|தெரு|சந்து|மாவட்டம்|மாநகராட்சி", re.I)
    bad = ("administrative", "populated_place", "residential", "postcode", "suburb", "neighbourhood", "neighborhood", "locality", "office", "hospital")
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        cats = [str(c).lower() for c in item.get("categories", [])]
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        allowed = any(c.startswith("tourism.") or "museum" in c or "culture" in c or "place_of_worship" in c or "historic" in c or "heritage" in c or c.startswith("natural") or "park" in c for c in cats)
        if name and key not in seen and not deny.search(name) and not any(any(b in c for b in bad) for c in cats) and allowed:
            seen.add(key)
            out.append(item)
    return out


def clean_restaurants(items):
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        cats = [str(c).lower() for c in item.get("categories", [])]
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if name and key not in seen and any(c.startswith("catering.") for c in cats):
            seen.add(key)
            out.append(item)
    return out


def trip_price_snapshot(trip):
    services = trip.get("services", {})
    flight_prices = [x.get("price") for x in listv(services, "flights") if isinstance(x.get("price"), (int, float))]
    hotel_prices = [x.get("price") for x in listv(services, "hotels") if isinstance(x.get("price"), (int, float))]
    cheapest_flight = min(flight_prices) if flight_prices else None
    cheapest_hotel = min(hotel_prices) if hotel_prices else None
    nights = int(trip.get("request", {}).get("durationNights") or 0)
    subtotal = cheapest_flight + cheapest_hotel * nights if cheapest_flight is not None and cheapest_hotel is not None else None
    return cheapest_flight, cheapest_hotel, subtotal


def compare(trips):
    rows = [(label, *trip_price_snapshot(trip)) for label, trip in trips]
    if not rows:
        return "## 🔎 Comparison summary\n\nNo comparable live MCP data is available."
    header = "| Metric | " + " | ".join(r[0] for r in rows) + " |"
    separator = "|---|" + "---:|" * len(rows)
    flight_row = "| Cheapest flight | " + " | ".join(money(r[1]) if r[1] is not None else "—" for r in rows) + " |"
    hotel_row = "| Cheapest hotel/night | " + " | ".join(money(r[2]) if r[2] is not None else "—" for r in rows) + " |"
    subtotal_row = "| Live flight + hotel subtotal | " + " | ".join(money(r[3]) if r[3] is not None else "—" for r in rows) + " |"
    priced = [(r[0], r[3]) for r in rows if r[3] is not None]
    if priced:
        best_label, best_value = min(priced, key=lambda item: item[1])
        verdict = f"### 🏆 Best value: {best_label}\n\nBased strictly on returned live flight + hotel cost, **{best_label}** is the lowest-cost option at **{money(best_value)}**."
    else:
        verdict = "### ⚠️ No cost winner\n\nThe returned MCP data does not contain enough usable flight and hotel prices to rank these destinations."
    return "## 🔎 Comparison summary\n\n" + "\n".join([header, separator, flight_row, hotel_row, subtotal_row]) + "\n\n" + verdict


def render_trip(trip):
    req = trip.get("request", {})
    services = trip.get("services", {})
    live = trip.get("liveDataSummary", {})
    flights = listv(services, "flights")
    hotels = listv(services, "hotels")
    atts = clean_attractions(services.get("attractions"))
    rests = clean_restaurants(services.get("restaurants"))
    weather = services.get("weather") if isinstance(services.get("weather"), dict) else {}
    budget = services.get("budget") if isinstance(services.get("budget"), dict) else {}
    start, end = iso(req.get("departDate")), iso(req.get("returnDate"))
    nights = int(req.get("durationNights") or ((end - start).days if start and end else 0))
    days = nights + 1
    lines = ["## ✈️ Trip at a glance", "", f"**{req.get('origin','—')} → {req.get('destinationCity','—')}, {req.get('destinationCountry','')}**", f"**{req.get('departDate','—')} → {req.get('returnDate','—')} · {req.get('travelers',1)} traveler(s) · {nights} night(s) / {days} calendar day(s)**", f"Budget: **{req.get('budgetLevel','budget')}**", "", "## 🛫 Flights", ""]
    if flights:
        lines += ["| Airline | Price | Departure | Arrival | Duration | Stops |", "|---|---:|---|---|---:|---:|"]
        for flight in flights[:5]:
            stops = int(flight.get("stops", 0) or 0)
            lines.append(f"| {flight.get('airline','Unknown')} | {money(flight.get('price'),flight.get('currency','USD'))} | {flight.get('departure','—')} | {flight.get('arrival','—')} | {flight.get('duration','—')} | {'Non-stop' if stops == 0 else f'{stops} stop(s)'} |")
        lines.append("\n*Live provider results; prices and availability can change.*")
    else:
        error = services.get("flights", {}).get("error", "No live flight options were returned.") if isinstance(services.get("flights"), dict) else "No live flight options were returned."
        lines.append(f"**Live flights unavailable.** {error}")
    lines += ["", "## 🏨 Hotels", ""]
    if hotels:
        lines += ["| Hotel | Nightly | Rating | Reviews |", "|---|---:|---:|---:|"]
        for hotel in hotels[:6]:
            lines.append(f"| {hotel.get('name','Unknown')} | {money(hotel.get('price'),hotel.get('currency','USD'))} | {hotel.get('rating','—')} | {hotel.get('reviews','—')} |")
        lines.append("\n*Live hotel rates returned for the requested dates.*")
    else:
        error = services.get("hotels", {}).get("error", "No live hotel options were returned.") if isinstance(services.get("hotels"), dict) else "No live hotel options were returned."
        lines.append(f"**Live hotels unavailable.** {error}")
    lines += ["", "## 📍 Things to do", ""]
    lines += [f"- **{a.get('name')}" + (f"** — {a.get('description')}" if a.get('description') else "**") for a in atts[:8]] if atts else ["- No high-quality live tourist attractions were returned."]
    lines += ["", "## 🍽️ Food picks", ""]
    lines += [f"- **{r.get('name')}**" for r in rests[:8]] if rests else ["- No verified restaurant results were returned."]
    lines += ["", "## 🌦️ Weather", ""]
    rows = weather.get("results", []) if isinstance(weather.get("results"), list) else []
    if rows:
        lines += ["| Date | Temp | Feels like | Conditions | Rain |", "|---|---:|---:|---|---:|"]
        for row in rows:
            lines.append(f"| {row.get('date','—')} | {row.get('temperature','—')}°C | {row.get('feelsLike','—')}°C | {row.get('description','—')} | {row.get('precipitationProbability','—')}% |")
    else:
        lines.append(f"**Live weather unavailable.** {weather.get('error','No forecast data was returned for the requested dates.')}")
    lines += ["", "## 💰 Budget", ""]
    if budget:
        currency = budget.get("currency", "USD")
        breakdown = budget.get("breakdown", {})
        lines += [f"**Generic estimate:** {money(budget.get('total_budget'),currency)}", f"- Flights estimate: {money(breakdown.get('flights_estimate'),currency)}", f"- Accommodation estimate: {money(breakdown.get('accommodation_estimate'),currency)}", f"- Daily expenses estimate: {money(breakdown.get('daily_expenses_estimate'),currency)}", "", "*Generic planning estimate only — not a live booking total.*"]
    if live.get("complete"):
        lines += ["", f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'), live.get('currency','USD'))}"]
    lines += ["", "## 🗓️ Suggested itinerary", ""]
    if start and end:
        current, ai, ri = start, 0, 0
        while current <= end:
            day_no = (current - start).days + 1
            lines.append(f"### Day {day_no} · {current.isoformat()}")
            if day_no == 1:
                lines.append("- ✈️ Arrival / check-in")
            elif current == end:
                lines.append("- 🧳 Check-out / departure")
            else:
                picks = []
                if ai < len(atts): picks.append(atts[ai]); ai += 1
                if ai < len(atts) and len(picks) == 1 and day_no < days - 1: picks.append(atts[ai]); ai += 1
                if picks:
                    for j, attraction in enumerate(picks): lines.append(f"- **{'Morning' if j == 0 else 'Afternoon'}:** {attraction.get('name')}")
                else:
                    lines.append("- Keep this period flexible rather than inventing another attraction.")
            if rests:
                lines.append(f"- 🍽️ **Food:** {rests[ri % len(rests)].get('name')}")
                ri += 1
            lines.append("")
            current += timedelta(days=1)
    lines += ["## ⚠️ Notes", "", "- Live prices and availability can change before booking.", "- Fresh travel facts above were returned by the MCP service bundle.", "- Airport/city resolution for flights is handled server-side by MCP before SerpApi.", "- No itinerary item is invented when the live provider returns nothing."]
    return "\n".join(lines)


async def answer_from_stored(question):
    trips = st.session_state.trip_collection
    if not trips:
        return "## 🧭 No stored trip data yet\n\nStart with a complete trip request first."
    compact = {
        city: {
            "request": trip.get("request", {}),
            "flights": listv(trip.get("services", {}), "flights")[:10],
            "hotels": listv(trip.get("services", {}), "hotels")[:10],
            "budget": trip.get("services", {}).get("budget", {}),
        }
        for city, trip in trips.items()
    }
    prompt = f"""Answer the user's follow-up using ONLY the stored MCP data below. Do not invent or estimate facts. If the data is insufficient, say so.\n\nSTORED MCP DATA:\n{json.dumps(compact, ensure_ascii=False)[:9000]}\n\nQUESTION:\n{question}"""
    result = await model(max_tokens=300).ainvoke([HumanMessage(content=prompt)])
    return (result.content or "").strip() or "I could not answer that from the stored MCP data."


async def open_mcp():
    stack = AsyncExitStack()
    transport = await stack.enter_async_context(sse_client(server_url))
    session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
    return stack, session


async def fetch_parallel(session, requests, status):
    async def one(index, request):
        try:
            return index, await mcp_trip(session, request), None
        except Exception as exc:
            return index, None, f"{type(exc).__name__}: {exc}"
    results = await asyncio.gather(*(one(i, request) for i, request in enumerate(requests)))
    results.sort(key=lambda x: x[0])
    return results


def build_requests(route: RouterDecision):
    origin = normalize_city(route.origin)
    requests = []
    for destination in route.destinations:
        city = normalize_city(destination.city)
        if not city:
            continue
        requests.append({
            "origin": origin,
            "destinationCity": city,
            "destinationCountry": destination.country or "",
            "departDate": route.departDate,
            "returnDate": route.returnDate,
            "passengers": int(route.passengers or 1),
            "budgetLevel": route.budgetLevel or "budget",
            "placesRadius": 5000,
        })
    return requests


async def run_turn(user_message, holder):
    stack = None
    try:
        holder.info("🧠 Understanding your request…")
        route = await route_request()

        if route.action == "ASK":
            return "## 🧭 I need a little more information\n\n" + (route.missing or "Please provide origin, destination, departure date, return date, and traveler count.")

        if route.action == "REUSE":
            holder.info("♻️ Reusing previously fetched MCP data…")
            return await answer_from_stored(route.question or user_message)

        requests = build_requests(route)
        validation_errors = []
        valid_requests = []
        for request in requests:
            ok, error = validate_request(request)
            if ok:
                valid_requests.append(request)
            else:
                validation_errors.append(f"**{request.get('destinationCity','Trip')}:** {error}")

        if not valid_requests:
            return "## ⚠️ Cannot plan\n\n" + ("\n\n".join(validation_errors) if validation_errors else "Please provide a destination and valid future dates.")

        holder.info(f"⚡ Fetching live MCP data for {len(valid_requests)} destination(s)…")
        stack, session = await open_mcp()
        results = await fetch_parallel(session, valid_requests, holder)

        fetched = []
        rendered = []
        for index, trip, error in results:
            city = valid_requests[index]["destinationCity"]
            if error:
                rendered.append(f"## ❌ {city}\n\nMCP live-data call failed: `{error}`")
                continue
            fetched.append(trip)
            st.session_state.trip_collection[city.lower()] = trip

        if not fetched:
            return "## ⚠️ No live MCP results were returned."

        context = st.session_state.active_trip
        if route.replace_active or context is None:
            st.session_state.active_trip = fetched[0]

        for trip in fetched:
            rendered.append(render_trip(trip))

        if route.compare_with_active and context is not None:
            pool = [(context.get("request", {}).get("destinationCity", "Active trip"), context)]
            pool.extend((trip.get("request", {}).get("destinationCity", "Candidate"), trip) for trip in fetched)
            rendered.append(compare(pool))
        elif len(fetched) > 1:
            rendered.append(compare([(trip.get("request", {}).get("destinationCity", "Trip"), trip) for trip in fetched]))

        if validation_errors:
            rendered.insert(0, "## ⚠️ Some requests were not run\n\n" + "\n\n".join(validation_errors))

        holder.empty()
        return "\n\n---\n\n".join(rendered)

    except Exception as exc:
        holder.empty()
        return f"## ❌ Something went wrong\n\n`{type(exc).__name__}: {exc}`"
    finally:
        if stack is not None:
            await stack.aclose()


for chat_message in st.session_state.messages:
    with st.chat_message(chat_message["role"]):
        st.markdown(chat_message["content"])

if prompt := st.chat_input("Try: Chennai → Madurai, Aug 20–25, 1 traveler"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = asyncio.run(run_turn(prompt, st.empty()))
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
