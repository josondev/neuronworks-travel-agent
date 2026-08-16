import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from datetime import date, datetime, timedelta
from typing import Any

import nest_asyncio
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from mcp import ClientSession
from mcp.client.sse import sse_client

nest_asyncio.apply()
st.set_page_config(page_title="Neuronworks Travel Agent", page_icon="✈️", layout="wide")

st.markdown("""
<style>
:root{--bg:#080b14;--border:rgba(255,255,255,.10)}
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
    st.caption("GPT-OSS 20B router only · no NVIDIA")
    st.caption("Fresh travel facts always come from MCP")
    st.caption("Airport/city resolution is handled by the MCP server")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_trip" not in st.session_state:
    st.session_state.active_trip = None
if "trip_collection" not in st.session_state:
    st.session_state.trip_collection = {}
if "comparison_trips" not in st.session_state:
    st.session_state.comparison_trips = {}


def model(max_tokens=500):
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=max_tokens)


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


def safe_json(raw):
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


MAX_ROUTER_TURNS = 10


def format_conversation(chat_history):
    recent = [m for m in chat_history if m["role"] in ("user", "assistant")][-MAX_ROUTER_TURNS:]
    lines = []
    for item in recent:
        speaker = "User" if item["role"] == "user" else "Assistant"
        content = item["content"] if speaker == "User" else item["content"][:400]
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines) if lines else "(no prior messages)"


def active_trip_summary(context):
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


def build_router_prompt(context, chat_history):
    return f"""You are an expert, factual AI Travel Agent and the semantic routing brain of a live MCP travel agent. Your goal is to plan realistic, bookable trips using ONLY real-time data returned by MCP.

### CURRENT CONTEXT
- Today's Date: 2026-08-16
- Relative dates such as next Friday or in 2 days must be resolved relative to today's date.

### ZERO-HALLUCINATION / MCP RULES
- MCP is the ONLY source of fresh travel facts: flights, hotels, places, restaurants, weather, budget, and currency.
- Never invent prices, availability, hotels, flights, attractions, weather, or airport codes.
- Exact live prices returned by MCP must be preserved.
- Generic budget output from MCP is a planning estimate, not a live booking total.
- Airport/city resolution is the MCP server's responsibility. Return plain city names only. NEVER fabricate IATA codes.
- Preserve origin, dates, traveler count, and budget from earlier turns when the user implies they remain unchanged.

### FOLLOW-UP / CONTEXT RULES
- A NEW destination always requires fresh MCP data.
- "Do the same", "same trip", "repeat this", "make the same plan", and similar wording must be interpreted from the full conversation, not from fixed keyword templates.
- If multiple new destinations are requested, include ALL of them in destinations so the application fetches them independently and in parallel.
- Comparisons with a new destination MUST fetch that candidate with MCP. Never replay the active trip as the candidate.
- REUSE is allowed only when the question can be answered completely from already-fetched MCP data.
- Comparisons and batch searches must not replace the active trip unless the user explicitly switches to a new active destination.

### COMPARISON / BEST VALUE
- compare_with_active=true when the user wants a new destination compared with the current active trip.
- replace_active=true only for a new primary plan or an explicit destination switch.
- Best value is a price-based conclusion from live flight + hotel subtotal unless the user explicitly asks for another criterion.

### ACTIONS
- FETCH: fetch live MCP data for one or more destinations.
- REUSE: answer using already-fetched MCP data; do not fetch new travel data.
- ASK: required information is missing and cannot be recovered from conversation context.

### REQUIRED ROUTER OUTPUT
Return ONLY one valid JSON object. No Markdown, no code fences, no explanation.
Use exactly this schema:
{
  "action": "FETCH | REUSE | ASK",
  "missing": null,
  "question": null,
  "origin": null,
  "destinations": [],
  "departDate": null,
  "returnDate": null,
  "passengers": null,
  "budgetLevel": null,
  "replace_active": false,
  "compare_with_active": false
}

FETCH examples:
- First request for Chennai to Madurai => action FETCH, origin Chennai, destinations [Madurai], replace_active true.
- compare this with Coimbatore => action FETCH, destinations [Coimbatore], compare_with_active true, replace_active false.
- do the same for Madurai, Kodaikanal and Ooty => action FETCH, destinations [Madurai, Kodaikanal, Ooty].
- Preserve earlier origin/dates/passengers/budget when not restated.

REUSE examples:
- Which hotel is cheapest? when that hotel's live data is already stored.
- What was the cheapest flight? when flight data is already stored.
In REUSE, set question to the user's actual question.

ASK:
- Use ASK only when origin, destination, dates, or traveler count truly cannot be recovered from the full conversation or active context.

### USER-FACING OUTPUT PRINCIPLES
1. Summary: concise breakdown of live flight and hotel options.
2. Itinerary: day-by-day plan using only provider-returned attractions and restaurants.
3. Budget: generic MCP estimate plus separate live-data subtotal when available.
4. Disclaimer: Prices and availability are subject to change.
5. If live data is missing, state that plainly rather than inventing a substitute.


ACTIVE TRIP CONTEXT:
{active_trip_summary(context)}

RECENT CONVERSATION:
{format_conversation(chat_history)}
"""


async def route_request(context, chat_history):
    try:
        result = await model(max_tokens=500).ainvoke([
            HumanMessage(content=build_router_prompt(context, chat_history))
        ])
        parsed = safe_json(result.content)
        if isinstance(parsed, dict) and parsed.get("action") in {"FETCH", "REUSE", "ASK"}:
            return parsed
    except Exception:
        pass
    return {"action": "ASK", "missing": "Please provide origin, destination, departure date, return date, and traveler count."}


def validate_request(req):
    start, end = iso(req.get("departDate")), iso(req.get("returnDate"))
    if not req.get("origin") or not req.get("destinationCity"):
        return False, "Please provide a valid origin and destination."
    if not start or not end:
        return False, "Please provide departure and return dates."
    if start < date.today():
        return False, f"The departure date {start.isoformat()} is in the past. Today is {date.today().isoformat()}."
    if end <= start:
        return False, "Return date must be after departure date."
    return True, ""


async def mcp_trip(session, request):
    result = await asyncio.wait_for(
        session.call_tool("build_trip_data", arguments=request),
        timeout=20,
    )
    if not result.content:
        raise RuntimeError("MCP returned no content")
    return json.loads(result.content[0].text)


def listv(services, key):
    value = services.get(key)
    return value if isinstance(value, list) else []


def clean_attractions(items):
    deny = re.compile(
        r"\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|bus\s*station|railway|parking|signal|flyover|underpass|bypass|overpass|salai|theru|nagar|colony|layout|township|extension|ward|sector|block|circle|chowk|hospital|water\s*works|car\s*shelter)\b|நகரம்|சாலை|தெரு|சந்து|மாவட்டம்|மாநகராட்சி",
        re.I,
    )
    bad = ("administrative", "populated_place", "residential", "postcode", "suburb", "neighbourhood", "neighborhood", "locality", "office", "hospital")
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        cats = [str(c).lower() for c in item.get("categories", [])]
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        allowed = any(
            c.startswith("tourism.") or "museum" in c or "culture" in c or "place_of_worship" in c
            or "historic" in c or "heritage" in c or c.startswith("natural") or "park" in c
            for c in cats
        )
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

    out = [
        "## ✈️ Trip at a glance", "",
        f"**{req.get('origin','—')} → {req.get('destinationCity','—')}, {req.get('destinationCountry','')}**",
        f"**{req.get('departDate','—')} → {req.get('returnDate','—')} · {req.get('travelers',req.get('passengers',1))} traveler(s) · {nights} night(s) / {days} calendar day(s)**",
        f"Budget: **{req.get('budgetLevel','budget')}**", "", "## 🛫 Flights", ""
    ]

    if flights:
        out += ["| Airline | Price | Departure | Arrival | Duration | Stops |", "|---|---:|---|---|---:|---:|"]
        for flight in flights[:5]:
            stops = int(flight.get("stops", 0) or 0)
            out.append(
                f"| {flight.get('airline','Unknown')} | {money(flight.get('price'),flight.get('currency','USD'))} | "
                f"{flight.get('departure','—')} | {flight.get('arrival','—')} | {flight.get('duration','—')} | "
                f"{'Non-stop' if stops == 0 else f'{stops} stop(s)'} |"
            )
    else:
        error = services.get("flights", {}).get("error", "No live flight options were returned.") if isinstance(services.get("flights"), dict) else "No live flight options were returned."
        out.append(f"**Live flights unavailable.** {error}")

    out += ["", "## 🏨 Hotels", ""]
    if hotels:
        out += ["| Hotel | Nightly | Rating | Reviews |", "|---|---:|---:|---:|"]
        for hotel in hotels[:6]:
            out.append(
                f"| {hotel.get('name','Unknown')} | {money(hotel.get('price'),hotel.get('currency','USD'))} | "
                f"{hotel.get('rating','—')} | {hotel.get('reviews','—')} |"
            )
    else:
        error = services.get("hotels", {}).get("error", "No live hotel options were returned.") if isinstance(services.get("hotels"), dict) else "No live hotel options were returned."
        out.append(f"**Live hotels unavailable.** {error}")

    out += ["", "## 📍 Things to do", ""]
    out += [f"- **{a.get('name')}**" for a in atts[:8]] if atts else ["- No high-quality live tourist attractions were returned."]

    out += ["", "## 🍽️ Food picks", ""]
    out += [f"- **{r.get('name')}**" for r in rests[:8]] if rests else ["- No verified restaurant results were returned."]

    out += ["", "## 🌦️ Weather", ""]
    rows = weather.get("results", []) if isinstance(weather.get("results"), list) else []
    if rows:
        out += ["| Date | Temp | Feels like | Conditions | Rain |", "|---|---:|---:|---|---:|"]
        for row in rows:
            out.append(
                f"| {row.get('date','—')} | {row.get('temperature','—')}°C | {row.get('feelsLike','—')}°C | "
                f"{row.get('description','—')} | {row.get('precipitationProbability','—')}% |"
            )
    else:
        out.append(f"**Live weather unavailable.** {weather.get('error','No forecast data was returned for the requested dates.')}")

    out += ["", "## 💰 Budget", ""]
    if budget:
        currency = budget.get("currency", "USD")
        breakdown = budget.get("breakdown", {})
        out += [
            f"**Generic estimate:** {money(budget.get('total_budget'),currency)}",
            f"- Flights estimate: {money(breakdown.get('flights_estimate'),currency)}",
            f"- Accommodation estimate: {money(breakdown.get('accommodation_estimate'),currency)}",
            f"- Daily expenses estimate: {money(breakdown.get('daily_expenses_estimate'),currency)}",
            "",
            "*Generic planning estimate only — not a live booking total.*",
        ]
    if live.get("complete"):
        out += ["", f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'),live.get('currency','USD'))}"]

    out += ["", "## 🗓️ Suggested itinerary", ""]
    if start and end:
        current, attraction_index, restaurant_index = start, 0, 0
        while current <= end:
            day_number = (current - start).days + 1
            out.append(f"### Day {day_number} · {current.isoformat()}")
            if day_number == 1:
                out.append("- ✈️ Arrival / check-in")
            elif current == end:
                out.append("- 🧳 Check-out / departure")
            else:
                picks = []
                if attraction_index < len(atts):
                    picks.append(atts[attraction_index]); attraction_index += 1
                if attraction_index < len(atts) and len(picks) == 1 and day_number < days - 1:
                    picks.append(atts[attraction_index]); attraction_index += 1
                if picks:
                    for j, attraction in enumerate(picks):
                        out.append(f"- **{'Morning' if j == 0 else 'Afternoon'}:** {attraction.get('name')}")
                else:
                    out.append("- Keep this period flexible rather than inventing another attraction.")
            if rests:
                out.append(f"- 🍽️ **Food:** {rests[restaurant_index % len(rests)].get('name')}")
                restaurant_index += 1
            out.append("")
            current += timedelta(days=1)

    out += [
        "## ⚠️ Notes", "",
        "- Live prices and availability can change before booking.",
        "- Fresh travel facts above were returned by the MCP service bundle.",
        "- Airport/city resolution for flights is handled server-side by the MCP flight service.",
        "- No itinerary item is invented when the live provider returns nothing.",
    ]
    return "\n".join(out)


def trip_price_snapshot(trip):
    services = trip.get("services", {})
    flight_prices = [
        x.get("price") for x in listv(services, "flights")
        if isinstance(x.get("price"), (int, float))
    ]
    hotel_prices = [
        x.get("price") for x in listv(services, "hotels")
        if isinstance(x.get("price"), (int, float))
    ]
    cheapest_flight = min(flight_prices) if flight_prices else None
    cheapest_hotel = min(hotel_prices) if hotel_prices else None
    nights = int(trip.get("request", {}).get("durationNights") or 0)
    subtotal = (
        cheapest_flight + cheapest_hotel * nights
        if cheapest_flight is not None and cheapest_hotel is not None
        else None
    )
    return cheapest_flight, cheapest_hotel, subtotal


def compare(trips):
    rows = []
    for label, trip in trips:
        flight, hotel, subtotal = trip_price_snapshot(trip)
        rows.append((label, flight, hotel, subtotal))

    if not rows:
        return "## 🔎 Comparison summary\n\nNo comparable live trip data is available."

    header = "| Metric | " + " | ".join(label for label, *_ in rows) + " |"
    separator = "|---|" + "---:|" * len(rows)
    flight_row = "| Cheapest flight | " + " | ".join(money(v) if v is not None else "—" for _, v, _, _ in rows) + " |"
    hotel_row = "| Cheapest hotel/night | " + " | ".join(money(v) if v is not None else "—" for _, _, v, _ in rows) + " |"
    subtotal_row = "| Live flight + hotel subtotal | " + " | ".join(money(v) if v is not None else "—" for _, _, _, v in rows) + " |"

    priced = [(label, subtotal) for label, _, _, subtotal in rows if subtotal is not None]
    if priced:
        best_label, best_value = min(priced, key=lambda item: item[1])
        verdict = (
            f"### 🏆 Best value: {best_label}\n\n"
            f"Based strictly on the returned live flight + hotel subtotal, **{best_label}** is the lowest-cost option at **{money(best_value)}**."
        )
        if len(priced) < len(rows):
            verdict += " Some destinations were excluded because MCP did not return both a usable flight and hotel price."
    else:
        verdict = "### ⚠️ No cost winner\n\nThe returned MCP data does not contain enough usable flight and hotel prices to determine a best-value option."

    return "## 🔎 Comparison summary\n\n" + "\n".join([header, separator, flight_row, hotel_row, subtotal_row]) + f"\n\n{verdict}"


async def answer_from_context(question, context):
    payload = {
        "request": context.get("request", {}),
        "flights": listv(context.get("services", {}), "flights")[:10],
        "hotels": listv(context.get("services", {}), "hotels")[:10],
        "attractions": [a.get("name") for a in clean_attractions(context.get("services", {}).get("attractions"))[:10]],
        "restaurants": [r.get("name") for r in clean_restaurants(context.get("services", {}).get("restaurants"))[:10]],
        "weather": context.get("services", {}).get("weather", {}),
        "budget": context.get("services", {}).get("budget", {}),
        "liveDataSummary": context.get("liveDataSummary", {}),
    }
    prompt = f"""Answer the user's question using ONLY the previously fetched MCP data below. Do not invent facts, prices, places, weather, or availability. If the data is insufficient, say so plainly.

DATA:
{json.dumps(payload, ensure_ascii=False)[:7000]}

QUESTION:
{question}
"""
    try:
        result = await model(max_tokens=300).ainvoke([HumanMessage(content=prompt)])
        return (result.content or "").strip() or render_trip(context)
    except Exception:
        return render_trip(context)


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
    results.sort(key=lambda item: item[0])
    for index, trip, error in results:
        city = requests[index].get("destinationCity", "Trip")
        status.info(("⚠️ MCP failed" if error else "✅ MCP returned") + f": {city}")
    return results


def build_fetch_requests(route, context):
    base = dict((context or {}).get("request", {}))
    origin = route.get("origin") or base.get("origin")
    depart = route.get("departDate") or base.get("departDate")
    return_date = route.get("returnDate") or base.get("returnDate")
    passengers = int(route.get("passengers") or base.get("passengers") or base.get("travelers") or 1)
    budget = route.get("budgetLevel") or base.get("budgetLevel") or "budget"
    destinations = route.get("destinations") or []

    requests = []
    for destination in destinations:
        if isinstance(destination, dict):
            city = normalize_city(destination.get("city"))
            country = destination.get("country")
        else:
            city = normalize_city(destination)
            country = None
        if not city:
            continue

        # Compatibility with the current MCP schema: the backend now treats
        # these fields as city/location inputs and resolves the actual airport
        # server-side before SerpApi flight search.
        requests.append({
            "origin": normalize_city(origin),
            "destinationCity": city,
            "destinationCountry": country or "",
            "departDate": depart,
            "returnDate": return_date,
            "passengers": passengers,
            "budgetLevel": budget,
        })

    return requests


async def run_turn(message, holder):
    stack = None
    try:
        holder.info("🧠 Understanding your request…")
        context = st.session_state.active_trip
        route = await route_request(context, st.session_state.messages)
        action = str(route.get("action", "ASK")).upper()

        if action == "ASK":
            return f"## 🧭 I need a little more information\n\n{route.get('missing') or 'Please provide origin, destination, dates, and traveler count.'}"

        if action == "REUSE":
            if not context:
                return "## 🧭 No active trip yet\n\nStart with a complete trip request first."
            return await answer_from_context(route.get("question") or message, context)

        requests = build_fetch_requests(route, context)
        if not requests:
            return "## 🧭 I need a little more information\n\nPlease provide at least one destination."

        validation_errors = []
        valid_requests = []
        for request in requests:
            ok, error = validate_request(request)
            if ok:
                valid_requests.append(request)
            else:
                validation_errors.append(f"**{request.get('destinationCity', 'Trip')}:** {error}")

        if not valid_requests:
            return "## ⚠️ Cannot plan\n\n" + "\n\n".join(validation_errors)

        holder.info(f"⚡ Fetching live MCP data for {len(valid_requests)} destination(s)…")
        stack, session = await open_mcp()
        results = await fetch_parallel(session, valid_requests, holder)

        fetched = []
        rendered = []
        for index, trip, error in results:
            city = valid_requests[index].get("destinationCity", "Trip")
            if error:
                rendered.append(f"## ❌ {city}\n\nMCP live-data call failed: `{error}`")
                continue
            fetched.append(trip)
            st.session_state.trip_collection[city.lower()] = trip

        if not fetched:
            return "## ⚠️ No live MCP results were returned."

        replace_active = bool(route.get("replace_active")) or context is None
        compare_with_active = bool(route.get("compare_with_active")) and context is not None

        if replace_active:
            st.session_state.active_trip = fetched[0]

        if compare_with_active and context is not None:
            for trip in fetched:
                city = trip.get("request", {}).get("destinationCity", "candidate")
                st.session_state.comparison_trips[city.lower()] = trip

        for trip in fetched:
            rendered.append(render_trip(trip))

        if compare_with_active and context is not None:
            pool = [(context.get("request", {}).get("destinationCity", "Active trip"), context)]
            pool.extend(
                (trip.get("request", {}).get("destinationCity", "Candidate"), trip)
                for trip in fetched
            )
            rendered.append(compare(pool))
        elif len(fetched) > 1:
            # A batch of fresh destinations is also directly comparable.
            pool = [
                (trip.get("request", {}).get("destinationCity", "Trip"), trip)
                for trip in fetched
            ]
            rendered.append(compare(pool))

        if validation_errors:
            rendered.insert(0, "## ⚠️ Some requests were not run\n\n" + "\n\n".join(validation_errors))

        return "\n\n---\n\n".join(rendered)

    except Exception as exc:
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
