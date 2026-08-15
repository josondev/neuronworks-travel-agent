import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from datetime import date, datetime, timedelta

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
:root{--bg:#080b14;--border:rgba(255,255,255,.10);--text:#e5e7eb;--muted:#94a3b8}
.stApp{background:radial-gradient(circle at 10% 0%,rgba(37,99,235,.45),transparent 34%),radial-gradient(circle at 90% 10%,rgba(124,58,237,.40),transparent 32%),var(--bg)}
.block-container{max-width:1180px;padding-top:3rem;padding-bottom:6rem}
.hero{padding:26px 28px;border-radius:22px;margin-bottom:18px;background:linear-gradient(135deg,rgba(37,99,235,.26),rgba(124,58,237,.22));border:1px solid var(--border)}
.hero h1{margin:0;color:#fff;font-size:2.2rem}.hero p{margin:8px 0;color:#cbd5e1}
.pill{display:inline-block;padding:5px 11px;border-radius:999px;background:rgba(255,255,255,.09);color:#e2e8f0;font-size:.75rem;border:1px solid var(--border)}
div[data-testid="stChatMessage"]{border:1px solid var(--border);border-radius:18px;padding:1rem 1.1rem;margin:.7rem 0;background:rgba(15,23,42,.82)}
.muted{color:var(--muted);font-size:.82rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><span class="pill">● LIVE MCP · FAST MODE</span><h1>✈️ Neuronworks Travel Agent</h1><p>Live flights · hotels · places · restaurants · weather · budget · currency</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    server_url = st.text_input("MCP Server URL", "https://neuronworks-travel-agent.onrender.com/sse")
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    if not groq_api_key:
        st.warning("Enter GROQ_API_KEY.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key
    st.success("🟢 Fast mode ready")
    st.caption("Single model: openai/gpt-oss-20b on Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_trip" not in st.session_state:
    st.session_state.active_trip = None
if "comparison_trips" not in st.session_state:
    st.session_state.comparison_trips = {}
if "batch_trips" not in st.session_state:
    st.session_state.batch_trips = {}

# Known aliases are only a fast path. Unknown destinations are resolved by the router.
IATA = {
    "chennai":"MAA", "madras":"MAA", "madurai":"IXM", "coimbatore":"CJB", "colombo":"CMB",
    "bangalore":"BLR", "bengaluru":"BLR", "hyderabad":"HYD", "delhi":"DEL", "new delhi":"DEL",
    "mumbai":"BOM", "bombay":"BOM", "kochi":"COK", "ooty":"CJB", "udhagamandalam":"CJB",
    "kodaikanal":"IXM"
}
COUNTRY = {
    "madurai":"India", "coimbatore":"India", "chennai":"India", "colombo":"Sri Lanka",
    "ooty":"India", "udhagamandalam":"India", "kodaikanal":"India"
}


def iso(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None


def money(value, currency="USD"):
    try:
        return f"{currency} {float(value):,.2f}"
    except Exception:
        return "Unavailable"


def parse_natural_dates(text):
    month_names = "January|February|March|April|May|June|July|August|September|October|November|December"
    found = re.findall(rf"\b(?:{month_names})\s+\d{{1,2}},?\s+\d{{4}}\b", text, flags=re.I)
    result = []
    for raw in found[:2]:
        try:
            result.append(datetime.strptime(raw.replace(",", ""), "%B %d %Y").date().isoformat())
        except ValueError:
            pass
    if len(result) == 2:
        return result
    iso_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    return iso_dates[:2] if len(iso_dates) >= 2 else []


def parse_requested_nights(text):
    match = re.search(r"\b(\d+)\s*[- ]?night(?:s)?\b", text.lower())
    return int(match.group(1)) if match else None


def normalize_request(request, base=None):
    data = dict(base or {})
    for key, value in (request or {}).items():
        if value not in (None, ""):
            data[key] = value

    city = str(data.get("destinationCity") or "").strip()
    origin = str(data.get("origin") or "").strip()
    airport = str(data.get("destinationAirport") or "").strip()

    data["origin"] = IATA.get(origin.lower(), origin.upper()) if origin else ""
    data["destinationAirport"] = IATA.get(airport.lower(), airport.upper()) if airport else IATA.get(city.lower(), "")
    data["destinationCountry"] = data.get("destinationCountry") or COUNTRY.get(city.lower(), "India")
    data["passengers"] = int(data.get("passengers") or 1)
    data["budgetLevel"] = data.get("budgetLevel") or "budget"
    data["placesRadius"] = int(data.get("placesRadius") or 5000)
    return data


def split_destination_list(text):
    """Extract destinations from natural batch phrases without hardcoding city names."""
    cleaned = text.strip().strip(".")
    cleaned = re.sub(r"\s+(?:please|thanks)$", "", cleaned, flags=re.I).strip()
    parts = re.split(r"\s*,\s*|\s+and\s+|\s*&\s*", cleaned, flags=re.I)
    destinations = []
    for part in parts:
        part = re.sub(r"^(?:the\s+)?(?:city\s+of\s+)", "", part.strip(), flags=re.I)
        part = part.strip(" .,-")
        if part:
            destinations.append(part.title())
    # de-duplicate while preserving order
    seen, result = set(), []
    for destination in destinations:
        key = destination.lower()
        if key not in seen:
            seen.add(key)
            result.append(destination)
    return result


def local_route(message, context):
    text = message.strip().lower()
    base = dict((context or {}).get("request", {}))

    if re.search(r"\bcheapest\s+(?:hotel|flight)\b", text):
        return {"action": "REUSE"}

    # Generic batch: "do the same for A, B and C" / "make the same plan for A and B".
    batch = re.search(
        r"\b(?:do|make)\s+(?:the\s+)?same(?:\s+(?:trip|plan|planning|itinerary))?\s+(?:with|for|in|to)\s+(.+)$",
        text,
        flags=re.I,
    )
    if batch:
        raw_targets = batch.group(1).strip().rstrip("?")
        destinations = split_destination_list(raw_targets)
        if len(destinations) >= 2:
            return {"action": "BATCH_UPDATE", "destinations": destinations}
        if len(destinations) == 1:
            destination = destinations[0]
            return {
                "action": "UPDATE",
                "destinationCity": destination,
                "destinationAirport": IATA.get(destination.lower()),
                "destinationCountry": COUNTRY.get(destination.lower()),
            }

    compare = re.search(
        r"\bcompare\b.*\b(?:with|vs|versus|to)\s+([a-zA-Z][\w\s.'-]*?)(?:\?|\.|$)",
        text,
    )
    if compare:
        city = compare.group(1).strip().title()
        return {
            "action": "COMPARE",
            "destinationCity": city,
            "destinationAirport": IATA.get(city.lower()),
            "destinationCountry": COUNTRY.get(city.lower()),
        }

    change = re.search(
        r"\b(?:change|switch|move)\s+(?:the\s+)?destination\s+(?:to|into)\s+([a-zA-Z][\w\s.'-]*?)(?:\?|\.|$)",
        text,
    )
    if change:
        city = change.group(1).strip().title()
        return {
            "action": "UPDATE",
            "destinationCity": city,
            "destinationAirport": IATA.get(city.lower()),
            "destinationCountry": COUNTRY.get(city.lower()),
        }

    route = re.search(
        r"\bfrom\s+([a-zA-Z][a-zA-Z .'-]*?)\s+to\s+([a-zA-Z][a-zA-Z .'-]*?)"
        r"(?=\s+(?:from|for|on|between|with)\b|\s*$)",
        text,
    )
    if route:
        origin = route.group(1).strip().title()
        city = route.group(2).strip().title()
        base.update({
            "origin": IATA.get(origin.lower(), origin),
            "destinationCity": city,
            "destinationAirport": IATA.get(city.lower(), ""),
            "destinationCountry": COUNTRY.get(city.lower(), base.get("destinationCountry", "India")),
        })

    dates = parse_natural_dates(text)
    if len(dates) == 2:
        base["departDate"], base["returnDate"] = dates

    travelers = re.search(r"\b(\d+)\s*(?:traveler|travellers|people|persons|adults?)\b", text)
    if travelers:
        base["passengers"] = int(travelers.group(1))
    elif re.search(r"\bfor\s+1\s+(?:traveler|person|adult)\b", text):
        base["passengers"] = 1

    requested_nights = parse_requested_nights(text)
    if requested_nights is not None:
        base["requestedNights"] = requested_nights

    if any(word in text for word in ("budget", "cheap", "minimum")):
        base["budgetLevel"] = "budget"
    elif any(word in text for word in ("luxury", "luxurious")):
        base["budgetLevel"] = "luxury"
    elif any(word in text for word in ("mid-range", "moderate")):
        base["budgetLevel"] = "mid-range"

    if not base.get("destinationCity"):
        destination = re.search(
            r"\b(?:to|for|in)\s+(?:the\s+)?(?:city\s+of\s+)?([a-zA-Z][a-zA-Z .'-]+?)"
            r"(?:\s+from\s+|\s+between\s+|\s+for\s+\d|\s+on\s+|\s*$)",
            text,
        )
        if destination:
            city = destination.group(1).strip().title()
            base.update({
                "destinationCity": city,
                "destinationAirport": IATA.get(city.lower(), ""),
                "destinationCountry": COUNTRY.get(city.lower(), "India"),
            })

    if base.get("origin") and base.get("destinationCity") and base.get("departDate") and base.get("returnDate"):
        return {"action": "PLAN", **base}
    return None


async def llm_route(message, context):
    prompt = f"""Return ONE line only in this exact format:
ACTION|ORIGIN_IATA|DESTINATION_CITY|DESTINATION_IATA|DEPART_DATE|RETURN_DATE|PASSENGERS|BUDGET

Current trip context: {json.dumps((context or {}).get('request', {}), ensure_ascii=False)}
User turn: {message}

Actions: PLAN, UPDATE, COMPARE, REUSE, ASK.
For a phrase like "do the same with/for X", preserve the current origin, dates, travelers and budget and change only X.
For "do the same for A, B and C", the application may handle the list outside this parser; do not invent extra destinations.
For "compare with X", preserve the current trip and create X as a comparison candidate.
Dates must be YYYY-MM-DD and must never be invented.
IATA RULE: every airport value must be an official uppercase 3-letter IATA code. If the requested destination has no practical commercial passenger airport, use the nearest practical airport and keep the requested destination city separate. Never put a city name in an airport field.
If a required value cannot be resolved confidently, leave that field empty rather than guessing.
"""
    model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=80)
    result = await model.ainvoke([HumanMessage(content=prompt)])
    raw = (result.content or "").strip().splitlines()[0] if result.content else ""
    parts = [part.strip() for part in raw.split("|", 7)]
    if len(parts) != 8:
        raise ValueError("Fast router returned invalid format")
    action, origin, city, airport, depart_date, return_date, passengers, budget = parts
    return {
        "action": action.upper(),
        "origin": origin or None,
        "destinationCity": city or None,
        "destinationAirport": airport or None,
        "departDate": depart_date or None,
        "returnDate": return_date or None,
        "passengers": int(passengers) if passengers.isdigit() else None,
        "budgetLevel": budget or None,
    }


def validate_trip(req):
    start, end = iso(req.get("departDate")), iso(req.get("returnDate"))
    today = date.today()
    if not req.get("origin") or not req.get("destinationCity"):
        return False, "Please provide a valid origin and destination."
    if not start or not end:
        return False, "Please provide departure and return dates."
    if start < today:
        return False, f"The departure date {start} is in the past. Today is {today}."
    if end <= start:
        return False, "Return date must be after departure date."
    requested_nights = req.get("requestedNights")
    actual_nights = (end - start).days
    if requested_nights is not None and int(requested_nights) != actual_nights:
        return False, (
            f"You specified {requested_nights} night(s), but {req['departDate']} → {req['returnDate']} "
            f"contains {actual_nights} night(s). Please correct the dates or the night count."
        )
    return True, ""


async def resolve_airport(city):
    known = IATA.get(city.lower())
    if known:
        return known
    resolver = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=20)
    result = await resolver.ainvoke([
        HumanMessage(content=(
            f"Return ONLY one official uppercase 3-letter IATA code for the nearest practical "
            f"commercial passenger airport serving {city}. No explanation."
        ))
    ])
    match = re.search(r"\b[A-Z]{3}\b", (result.content or "").upper())
    return match.group(0) if match else ""


async def get_trip(session, args):
    result = await asyncio.wait_for(
        session.call_tool("build_trip_data", arguments=args),
        timeout=17,
    )
    return json.loads(result.content[0].text)


def clean_places(items, restaurants=False):
    bad = re.compile(
        r"\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|bus\s*station|"
        r"railway|parking|signal|flyover|underpass|bypass|overpass|salai|theru|sandhu|"
        r"mawatha|marg|nagar|colony|water\s*works|car\s*shelter|hospital|ward)\b",
        re.I,
    )
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        categories = ",".join(str(c).lower() for c in item.get("categories", []))
        if not name or bad.search(name):
            continue
        category_list = categories.split(",")
        if restaurants and not any(c.startswith("catering.") for c in category_list):
            continue
        if not restaurants and not any(
            c.startswith("tourism.")
            or "museum" in c
            or "culture" in c
            or "place_of_worship" in c
            or "historic" in c
            or "heritage" in c
            or c.startswith("natural")
            or "park" in c
            for c in category_list
        ):
            continue
        if not restaurants and re.search(r"\b(statue|viewpoint|train|triangle|building)\b", name, re.I):
            if not any(k in categories for k in ("historic", "culture", "museum", "place_of_worship")):
                continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def render(trip):
    request = trip.get("request", {})
    services = trip.get("services", {})
    live = trip.get("liveDataSummary", {})
    flights = services.get("flights", []) if isinstance(services.get("flights"), list) else []
    hotels = services.get("hotels", []) if isinstance(services.get("hotels"), list) else []
    attractions = clean_places(services.get("attractions"))
    restaurants = clean_places(services.get("restaurants"), True)
    weather = services.get("weather", {}) if isinstance(services.get("weather"), dict) else {}
    budget = services.get("budget", {}) if isinstance(services.get("budget"), dict) else {}

    lines = [
        "## ✈️ Trip at a glance",
        "",
        f"**{request.get('origin')} → {request.get('destinationCity')}, {request.get('destinationCountry', '')}**",
        f"**{request.get('departDate')} → {request.get('returnDate')} · {request.get('passengers', 1)} traveler(s) · {request.get('durationNights')} night(s)**",
        f"Budget: **{request.get('budgetLevel', 'budget')}**",
        "",
        "## 🛫 Flights",
        "",
    ]
    if flights:
        lines += [
            "| Airline | Price | Departure | Arrival | Duration | Stops |",
            "|---|---:|---|---|---:|---:|",
        ]
        for flight in flights[:5]:
            stops = int(flight.get("stops", 0) or 0)
            lines.append(
                f"| {flight.get('airline', 'Unknown')} | "
                f"{money(flight.get('price'), flight.get('currency', 'USD'))} | "
                f"{flight.get('departure', '—')} | {flight.get('arrival', '—')} | "
                f"{flight.get('duration', '—')} | "
                f"{'Non-stop' if stops == 0 else str(stops) + ' stop(s)'} |"
            )
    else:
        error = services.get("flights", {}).get("error", "No live flight results.") if isinstance(services.get("flights"), dict) else "No live flight results."
        lines.append(f"**Live flights unavailable.** {error}")

    lines += ["", "## 🏨 Hotels", ""]
    if hotels:
        lines += ["| Hotel | Nightly | Rating | Reviews |", "|---|---:|---:|---:|"]
        for hotel in hotels[:6]:
            lines.append(
                f"| {hotel.get('name', 'Unknown')} | {money(hotel.get('price'), hotel.get('currency', 'USD'))} | "
                f"{hotel.get('rating', '—')} | {hotel.get('reviews', '—')} |"
            )
    else:
        lines.append("**Live hotels unavailable.**")

    lines += ["", "## 📍 Things to do", ""]
    lines.extend(f"- **{place.get('name')}**" for place in attractions[:8])
    if not attractions:
        lines.append("- No high-quality live tourist attractions were returned.")

    lines += ["", "## 🍽️ Food picks", ""]
    lines.extend(f"- **{restaurant.get('name')}**" for restaurant in restaurants[:8])
    if not restaurants:
        lines.append("- No verified restaurant results were returned.")

    lines += ["", "## 🌦️ Weather", ""]
    weather_rows = weather.get("results", []) if isinstance(weather.get("results"), list) else []
    if weather_rows:
        for row in weather_rows:
            lines.append(
                f"- **{row.get('date', '—')}:** {row.get('temperature', '—')}°C, "
                f"feels like {row.get('feelsLike', '—')}°C, {row.get('description', '—')}, "
                f"rain {row.get('precipitationProbability', '—')}%"
            )
    else:
        lines.append(f"- {weather.get('error', 'No live weather coverage returned.')}")

    lines += ["", "## 💰 Budget", ""]
    if budget:
        breakdown = budget.get("breakdown", {})
        currency = budget.get("currency", "USD")
        lines += [
            f"**Generic estimate:** {money(budget.get('total_budget'), currency)}",
            f"- Flights: {money(breakdown.get('flights_estimate'), currency)}",
            f"- Accommodation: {money(breakdown.get('accommodation_estimate'), currency)}",
            f"- Daily expenses: {money(breakdown.get('daily_expenses_estimate'), currency)}",
        ]
    if live.get("complete"):
        lines.append(f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'), live.get('currency', 'USD'))}")

    lines += ["", "## 🗓️ Suggested itinerary", ""]
    start, end = iso(request.get("departDate")), iso(request.get("returnDate"))
    if start and end:
        attraction_index = 0
        current = start
        while current <= end:
            day_number = (current - start).days + 1
            lines.append(f"### Day {day_number} · {current.isoformat()}")
            if day_number == 1:
                lines.append("- ✈️ Arrive and check in")
            elif current == end:
                lines.append("- 🧳 Check-out / departure")
            else:
                chosen = attractions[attraction_index:attraction_index + 2]
                for place in chosen:
                    lines.append(f"- **Sightseeing:** {place.get('name')}")
                attraction_index += len(chosen)
                if not chosen:
                    lines.append("- No additional high-quality live attractions were returned for this day.")
            lines.append("")
            current += timedelta(days=1)

    lines += [
        "## ⚠️ Notes",
        "",
        "- Live prices and availability can change before booking.",
        "- Generic budget estimates are not live booking totals.",
        "- Weather is shown only for dates actually covered by the live provider.",
        "- Follow-up messages reuse the saved trip context.",
    ]
    return "\n".join(lines)


def compare_stats(current, candidate):
    def stats(trip):
        services = trip.get("services", {})
        request = trip.get("request", {})
        flights = services.get("flights", []) if isinstance(services.get("flights"), list) else []
        hotels = services.get("hotels", []) if isinstance(services.get("hotels"), list) else []
        cheapest_flight = min((float(x["price"]) for x in flights if isinstance(x.get("price"), (int, float))), default=None)
        cheapest_hotel = min((float(x["price"]) for x in hotels if isinstance(x.get("price"), (int, float))), default=None)
        nights = int(request.get("durationNights") or 0)
        subtotal = cheapest_flight + cheapest_hotel * nights if cheapest_flight is not None and cheapest_hotel is not None else None
        return {
            "flight": cheapest_flight,
            "hotel": cheapest_hotel,
            "subtotal": subtotal,
            "flights": len(flights),
            "hotels": len(hotels),
            "attractions": len(clean_places(services.get("attractions"))),
            "restaurants": len(clean_places(services.get("restaurants"), True)),
        }

    a, b = stats(current), stats(candidate)
    current_name = current["request"].get("destinationCity")
    candidate_name = candidate["request"].get("destinationCity")
    lines = [
        f"## 🔎 {current_name} vs {candidate_name}",
        "",
        f"**Same trip context:** {current['request'].get('origin')} · {current['request'].get('departDate')} → {current['request'].get('returnDate')} · {current['request'].get('passengers', 1)} traveler(s) · {current['request'].get('budgetLevel', 'budget')}",
        "",
        "| Metric | Current | Candidate |",
        "|---|---:|---:|",
    ]
    rows = [
        ("Cheapest flight", a["flight"], b["flight"], True),
        ("Cheapest hotel/night", a["hotel"], b["hotel"], True),
        ("Flight options", a["flights"], b["flights"], False),
        ("Hotel options", a["hotels"], b["hotels"], False),
        ("Verified attractions", a["attractions"], b["attractions"], False),
        ("Restaurants", a["restaurants"], b["restaurants"], False),
    ]
    for label, left, right, is_money in rows:
        left_value = money(left) if is_money and left is not None else ("—" if left is None else str(left))
        right_value = money(right) if is_money and right is not None else ("—" if right is None else str(right))
        lines.append(f"| {label} | {left_value} | {right_value} |")
    lines += ["", "### Recommendation"]
    if a["subtotal"] is not None and b["subtotal"] is not None:
        winner = current_name if a["subtotal"] < b["subtotal"] else candidate_name
        lines.append(f"- Lower live flight + hotel subtotal: **{winner}** ({money(min(a['subtotal'], b['subtotal']))}).")
    lines.append("- This comparison does not replace your active trip context.")
    return "\n".join(lines)


def render_batch(trips):
    sections = []
    for city, trip in trips.items():
        sections.append(f"# 📍 {city}\n\n{render(trip)}")
    return "\n\n---\n\n".join(sections)


async def run_turn(message, placeholder):
    async with AsyncExitStack() as stack:
        placeholder.info("⚡ Fast mode…")
        try:
            context = st.session_state.active_trip
            route = local_route(message, context)
            if route is None:
                placeholder.info("⚡ Fast router fallback…")
                route = await llm_route(message, context)

            action = str(route.get("action", "ASK")).upper()
            base = (context or {}).get("request", {})

            if action == "REUSE":
                if not context:
                    return "## 🧭 No active trip\n\nStart with a complete trip request."
                lower = message.lower()
                services = context.get("services", {})
                if "cheapest hotel" in lower:
                    hotels = [h for h in services.get("hotels", []) if isinstance(h.get("price"), (int, float))]
                    if hotels:
                        hotel = min(hotels, key=lambda x: x["price"])
                        return f"### 🏨 Cheapest hotel\n\n**{hotel.get('name')}** — {money(hotel.get('price'), hotel.get('currency', 'USD'))}/night."
                if "cheapest flight" in lower:
                    flights = [f for f in services.get("flights", []) if isinstance(f.get("price"), (int, float))]
                    if flights:
                        flight = min(flights, key=lambda x: x["price"])
                        return f"### 🛫 Cheapest flight\n\n**{flight.get('airline')}** — {money(flight.get('price'), flight.get('currency', 'USD'))}."
                return render(context)

            if action == "BATCH_UPDATE":
                if not context:
                    return "## 🧭 No active trip\n\nStart with one complete trip first, then ask for multiple destinations."

                requests = []
                for destination in route.get("destinations", []):
                    candidate = normalize_request({
                        "destinationCity": destination,
                        "destinationAirport": IATA.get(destination.lower()),
                        "destinationCountry": COUNTRY.get(destination.lower(), base.get("destinationCountry")),
                    }, base)
                    requests.append(candidate)

                # Resolve all missing airports concurrently. This remains fast for batches.
                missing = [req for req in requests if not req.get("destinationAirport")]
                if missing:
                    placeholder.info("⚡ Resolving airports for destinations…")
                    resolved = await asyncio.gather(*(resolve_airport(req["destinationCity"]) for req in missing), return_exceptions=True)
                    for req, airport in zip(missing, resolved):
                        if isinstance(airport, str):
                            req["destinationAirport"] = airport

                for req in requests:
                    ok, error = validate_trip(req)
                    if not ok:
                        return f"## ⚠️ Cannot plan {req.get('destinationCity')}\n\n{error}"
                    if not req.get("destinationAirport"):
                        return f"## ⚠️ Cannot find a reliable commercial airport for {req.get('destinationCity')}."

                transport = await stack.enter_async_context(sse_client(server_url))
                session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
                placeholder.info(f"⚡ Fetching {len(requests)} destinations in parallel…")
                results = await asyncio.gather(*(get_trip(session, req) for req in requests), return_exceptions=True)
                batch = {}
                failures = []
                for req, result in zip(requests, results):
                    city = req.get("destinationCity", "Unknown")
                    if isinstance(result, Exception):
                        failures.append(f"- **{city}:** {type(result).__name__}: {result}")
                        continue
                    if result.get("planningBlocked"):
                        failures.append(f"- **{city}:** {result.get('error', 'Trip could not be planned.')}" )
                        continue
                    batch[city] = result

                st.session_state.batch_trips = batch
                placeholder.info("✅ Batch planning complete")
                output = render_batch(batch) if batch else "## ❌ No destination could be planned."
                if failures:
                    output += "\n\n## ⚠️ Destinations with errors\n\n" + "\n".join(failures)
                return output

            request = normalize_request(route, base)

            # Resolve missing commercial airport generically, not just for a fixed city list.
            if request.get("destinationCity") and not request.get("destinationAirport"):
                placeholder.info(f"⚡ Resolving flight airport for {request['destinationCity']}…")
                request["destinationAirport"] = await resolve_airport(request["destinationCity"])

            if action == "UPDATE" and not context:
                action = "PLAN"

            ok, error = validate_trip(request)
            if not ok:
                return f"## ⚠️ {error}"

            if action == "COMPARE":
                if not context:
                    return "## 🧭 No active trip\n\nStart with a complete trip before comparing destinations."
                transport = await stack.enter_async_context(sse_client(server_url))
                session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
                placeholder.info(f"⚡ Fetching live {request.get('destinationCity')} data…")
                candidate = await get_trip(session, request)
                st.session_state.comparison_trips[request.get("destinationCity", "Unknown")] = candidate
                return compare_stats(context, candidate)

            if not request.get("destinationCity") or not request.get("origin"):
                return "## 🧭 I need a little more information\n\nPlease provide the origin and destination."

            transport = await stack.enter_async_context(sse_client(server_url))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            placeholder.info(f"⚡ Fetching live {request.get('destinationCity')} data…")
            trip = await get_trip(session, request)
            if trip.get("planningBlocked"):
                return f"## ⚠️ Trip cannot be planned\n\n{trip.get('error', 'Unknown error')}"

            # UPDATE/PLAN replaces the active trip. BATCH and COMPARE do not.
            st.session_state.active_trip = trip
            return render(trip)

        except Exception as exc:
            return f"## ❌ Something went wrong\n\n`{type(exc).__name__}: {exc}`"


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Try: Chennai → Madurai, Aug 20–25, 1 traveler"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = asyncio.run(run_turn(prompt, st.empty()))
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
