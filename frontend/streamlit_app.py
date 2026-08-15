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

st.set_page_config(
    page_title="Neuronworks Travel Agent",
    page_icon="✈️",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root{--bg:#080b14;--border:rgba(255,255,255,.10);--muted:#94a3b8}
    .stApp{background:
      radial-gradient(circle at 10% 0%,rgba(37,99,235,.42),transparent 34%),
      radial-gradient(circle at 90% 10%,rgba(124,58,237,.36),transparent 32%),
      var(--bg)}
    .block-container{max-width:1180px;padding-top:3rem;padding-bottom:6rem}
    .hero{padding:26px 28px;border-radius:22px;margin-bottom:18px;
      background:linear-gradient(135deg,rgba(37,99,235,.26),rgba(124,58,237,.22));
      border:1px solid var(--border)}
    .hero h1{margin:0;color:#fff;font-size:2.2rem}
    .hero p{margin:8px 0;color:#cbd5e1}
    .pill{display:inline-block;padding:5px 11px;border-radius:999px;
      background:rgba(255,255,255,.09);color:#e2e8f0;font-size:.75rem;
      border:1px solid var(--border)}
    div[data-testid="stChatMessage"]{border:1px solid var(--border);
      border-radius:18px;padding:1rem 1.1rem;margin:.7rem 0;background:rgba(15,23,42,.82)}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <span class="pill">● LIVE MCP · FAST MODE</span>
      <h1>✈️ Neuronworks Travel Agent</h1>
      <p>Live flights · hotels · places · restaurants · weather · budget · currency</p>
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
        "Groq API Key", type="password"
    )
    if not groq_api_key:
        st.warning("Enter GROQ_API_KEY.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key
    st.success("🟢 Fast mode ready")
    st.caption("Single model: openai/gpt-oss-20b on Groq")
    st.caption("No NVIDIA dependency")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_trip" not in st.session_state:
    st.session_state.active_trip = None
if "stored_trips" not in st.session_state:
    st.session_state.stored_trips = {}
if "comparison_trips" not in st.session_state:
    st.session_state.comparison_trips = {}

# Fast-path aliases only. Unknown destinations are resolved by one Groq call.
IATA = {
    "chennai": "MAA",
    "madras": "MAA",
    "madurai": "IXM",
    "coimbatore": "CJB",
    "colombo": "CMB",
    "bangalore": "BLR",
    "bengaluru": "BLR",
    "hyderabad": "HYD",
    "delhi": "DEL",
    "new delhi": "DEL",
    "mumbai": "BOM",
    "bombay": "BOM",
    "kochi": "COK",
    "ooty": "CJB",
    "udhagamandalam": "CJB",
    "kodaikanal": "IXM",
}

COUNTRY = {
    "madurai": "India",
    "coimbatore": "India",
    "chennai": "India",
    "colombo": "Sri Lanka",
    "ooty": "India",
    "udhagamandalam": "India",
    "kodaikanal": "India",
}


def iso(value: Any):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None


def money(value: Any, currency: str = "USD") -> str:
    try:
        return f"{currency} {float(value):,.2f}"
    except Exception:
        return "Unavailable"


def parse_natural_dates(text: str):
    month_names = (
        "January|February|March|April|May|June|July|August|September|"
        "October|November|December"
    )
    matches = re.findall(
        rf"\b(?:{month_names})\s+\d{{1,2}},?\s+\d{{4}}\b",
        text,
        flags=re.I,
    )
    result = []
    for raw in matches[:2]:
        try:
            result.append(
                datetime.strptime(
                    raw.replace(",", ""), "%B %d %Y"
                ).date().isoformat()
            )
        except ValueError:
            continue
    if len(result) == 2:
        return result
    iso_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    return iso_dates[:2] if len(iso_dates) >= 2 else []


def parse_requested_nights(text: str):
    match = re.search(r"\b(\d+)\s*[- ]?night(?:s)?\b", text.lower())
    return int(match.group(1)) if match else None


def normalize_request(req: dict, base: dict | None = None) -> dict:
    data = dict(base or {})
    for key, value in (req or {}).items():
        if value not in (None, ""):
            data[key] = value

    city = str(data.get("destinationCity") or "").strip()
    origin = str(data.get("origin") or "").strip()
    airport = str(data.get("destinationAirport") or "").strip()

    data["origin"] = IATA.get(origin.lower(), origin.upper()) if origin else ""
    data["destinationAirport"] = (
        IATA.get(airport.lower(), airport.upper())
        if airport
        else IATA.get(city.lower(), "")
    )
    data["destinationCountry"] = data.get("destinationCountry") or COUNTRY.get(
        city.lower(), "India"
    )
    data["passengers"] = int(data.get("passengers") or 1)
    data["budgetLevel"] = data.get("budgetLevel") or "budget"
    data["placesRadius"] = int(data.get("placesRadius") or 5000)
    return data


def split_destination_list(text: str) -> list[str]:
    cleaned = text.strip().strip(" .")
    cleaned = re.sub(r"\s+(?:please|thanks)$", "", cleaned, flags=re.I)
    parts = re.split(r"\s*,\s*|\s+and\s+|\s*&\s*", cleaned, flags=re.I)
    output = []
    seen = set()
    for part in parts:
        part = re.sub(
            r"^(?:the\s+)?(?:city\s+of\s+)",
            "",
            part.strip(),
            flags=re.I,
        ).strip(" .,-")
        if not part:
            continue
        key = part.lower()
        if key not in seen:
            seen.add(key)
            output.append(part.title())
    return output


def extract_same_targets(text: str) -> list[str]:
    patterns = [
        r"\bdo\s+the\s+same(?:\s+(?:trip|plan|planning|itinerary|thing|thingy))?\s+(?:with|for|in|to)\s+(.+)$",
        r"\bmake\s+the\s+same(?:\s+(?:trip|plan|planning|itinerary))?\s+(?:with|for|in|to)\s+(.+)$",
        r"\brepeat\s+(?:the\s+same\s+)?(?:trip|plan|planning|itinerary)\s+(?:for|in|with|to)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), flags=re.I)
        if match:
            targets = split_destination_list(match.group(1).rstrip("?"))
            if targets:
                return targets
    return []


def local_route(message: str, context: dict | None):
    text = message.strip().lower()
    base = dict((context or {}).get("request", {}))

    if re.search(r"\bcheapest\s+(?:hotel|flight)\b", text):
        return {"action": "REUSE"}

    targets = extract_same_targets(text)
    if targets:
        if len(targets) > 1:
            return {"action": "BATCH_UPDATE", "destinations": targets}
        city = targets[0]
        return {
            "action": "UPDATE",
            "destinationCity": city,
            "destinationAirport": IATA.get(city.lower()),
            "destinationCountry": COUNTRY.get(city.lower()),
        }

    match = re.search(
        r"\bcompare\b.*\b(?:with|vs|versus|to)\s+([a-zA-Z][\w\s.'-]*?)(?:\?|\.|$)",
        text,
    )
    if match:
        city = match.group(1).strip().title()
        return {
            "action": "COMPARE",
            "destinationCity": city,
            "destinationAirport": IATA.get(city.lower()),
            "destinationCountry": COUNTRY.get(city.lower()),
        }

    match = re.search(
        r"\b(?:change|switch|move)\s+(?:the\s+)?destination\s+(?:to|into)\s+"
        r"([a-zA-Z][\w\s.'-]*?)(?:\?|\.|$)",
        text,
    )
    if match:
        city = match.group(1).strip().title()
        return {
            "action": "UPDATE",
            "destinationCity": city,
            "destinationAirport": IATA.get(city.lower()),
            "destinationCountry": COUNTRY.get(city.lower()),
        }

    route = re.search(
        r"\bfrom\s+([a-zA-Z][a-zA-Z .'-]*?)\s+to\s+([a-zA-Z][a-zA-Z .'-]*?)"
        r"(?=\s+(?:from|for|on|between|with|with\s+the)\b|\s*$)",
        text,
    )
    if route:
        origin = route.group(1).strip().title()
        city = route.group(2).strip().title()
        base.update(
            {
                "origin": IATA.get(origin.lower(), origin),
                "destinationCity": city,
                "destinationAirport": IATA.get(city.lower(), ""),
                "destinationCountry": COUNTRY.get(
                    city.lower(), base.get("destinationCountry", "India")
                ),
            }
        )

    dates = parse_natural_dates(text)
    if len(dates) == 2:
        base["departDate"], base["returnDate"] = dates

    travelers = re.search(
        r"\b(\d+)\s*(?:traveler|travellers|people|persons|adults?)\b",
        text,
    )
    if travelers:
        base["passengers"] = int(travelers.group(1))
    elif re.search(r"\bfor\s+1\s+(?:traveler|person|adult)\b", text):
        base["passengers"] = 1

    nights = parse_requested_nights(text)
    if nights is not None:
        base["requestedNights"] = nights

    if any(word in text for word in ("budget", "cheap", "minimum")):
        base["budgetLevel"] = "budget"
    elif any(word in text for word in ("luxury", "luxurious")):
        base["budgetLevel"] = "luxury"
    elif any(word in text for word in ("mid-range", "moderate")):
        base["budgetLevel"] = "mid-range"

    if not base.get("destinationCity"):
        match = re.search(
            r"\b(?:to|for|in)\s+(?:the\s+)?(?:city\s+of\s+)?"
            r"([a-zA-Z][a-zA-Z .'-]+?)"
            r"(?:\s+from\s+|\s+between\s+|\s+for\s+\d|\s+on\s+|\s*$)",
            text,
        )
        if match:
            city = match.group(1).strip().title()
            base.update(
                {
                    "destinationCity": city,
                    "destinationAirport": IATA.get(city.lower(), ""),
                    "destinationCountry": COUNTRY.get(city.lower(), "India"),
                }
            )

    if (
        base.get("origin")
        and base.get("destinationCity")
        and base.get("departDate")
        and base.get("returnDate")
    ):
        return {"action": "PLAN", **base}
    return None


async def resolve_unknown_airports(cities: list[str]) -> dict[str, str]:
    unresolved = [city for city in cities if not IATA.get(city.lower())]
    mapping = {city.lower(): IATA[city.lower()] for city in cities if city.lower() in IATA}
    if not unresolved:
        return mapping

    prompt = f"""
Return ONLY one JSON object and nothing else.

Resolve practical flight-search airports for these destinations:
{json.dumps(unresolved, ensure_ascii=False)}

Rules:
- Use an official uppercase 3-letter IATA code.
- If a destination does not have a practical commercial passenger airport,
  use the nearest practical commercial passenger airport serving it.
- Never return a city name.
- Never invent a code.
Example: Ooty can use CJB; Kodaikanal can use IXM.
"""
    model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=120)
    result = await model.ainvoke([HumanMessage(content=prompt)])
    text = (result.content or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            raw = json.loads(match.group(0))
            if isinstance(raw, dict):
                for city, code in raw.items():
                    if isinstance(code, str) and re.fullmatch(r"[A-Z]{3}", code.strip()):
                        mapping[str(city).strip().lower()] = code.strip()
        except json.JSONDecodeError:
            pass
    return mapping


def validate_request(req: dict):
    start = iso(req.get("departDate"))
    end = iso(req.get("returnDate"))
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
        return (
            False,
            f"You specified {requested_nights} night(s), but {req['departDate']} → "
            f"{req['returnDate']} contains {actual_nights} night(s). "
            "Please correct the dates or the night count.",
        )
    return True, ""


async def get_trip(session: ClientSession, args: dict):
    result = await asyncio.wait_for(
        session.call_tool("build_trip_data", arguments=args),
        timeout=18,
    )
    return json.loads(result.content[0].text)


def clean_attractions(items: Any) -> list[dict]:
    tamil_locality = "நகரம்|சாலை|தெரு|சந்து|குறுக்குச்சாலை|மாவட்டம்|மாநகராட்சி"
    name_deny = re.compile(
        rf"(\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|"
        rf"bus\s*station|railway|parking|signal|flyover|underpass|bypass|"
        rf"overpass|salai|theru|sandhu|mawatha|marg|nagar|colony|layout|"
        rf"township|extension|ward|sector|block|circle|chowk|hospital|"
        rf"water\s*works|car\s*shelter)\b)|({tamil_locality})",
        re.I,
    )
    category_deny = (
        "administrative", "populated_place", "residential", "postcode",
        "suburb", "neighbourhood", "neighborhood", "locality",
        "commercial.building", "office", "hospital",
    )
    output, seen = [], set()

    for item in items if isinstance(items, list) else []:
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
            or "heritage" in c
            or c.startswith("natural")
            or "park" in c
            for c in categories
        )
        if not allowed:
            continue
        if re.search(r"\b(statue|viewpoint|train|triangle|building)\b", name, re.I):
            if not any(
                k in ",".join(categories)
                for k in ("historic", "culture", "museum", "place_of_worship", "heritage")
            ):
                continue
        seen.add(key)
        output.append(item)
    return output


def clean_restaurants(items: Any) -> list[dict]:
    name_deny = re.compile(
        r"\b(street|road|lane|mawatha|marg|salai|theru|sandhu|highway|"
        r"junction|bus\s*stop|station|nagar|colony)\b",
        re.I,
    )
    output, seen = [], set()
    for item in items if isinstance(items, list) else []:
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


def service_list(services: dict, key: str) -> list[dict]:
    value = services.get(key)
    return value if isinstance(value, list) else []


def render_trip(trip: dict, title_prefix: str = "") -> str:
    req = trip.get("request", {})
    services = trip.get("services", {})
    live = trip.get("liveDataSummary", {})

    flights = service_list(services, "flights")
    hotels = service_list(services, "hotels")
    attractions = clean_attractions(services.get("attractions"))
    restaurants = clean_restaurants(services.get("restaurants"))
    weather = services.get("weather") if isinstance(services.get("weather"), dict) else {}
    budget = services.get("budget") if isinstance(services.get("budget"), dict) else {}

    start = iso(req.get("departDate"))
    end = iso(req.get("returnDate"))
    nights = int(req.get("durationNights") or ((end - start).days if start and end else 0))
    days = nights + 1

    lines = [
        f"## ✈️ {title_prefix + ' ' if title_prefix else ''}Trip at a glance",
        "",
        f"**{req.get('origin', '—')} → {req.get('destinationCity', '—')}, {req.get('destinationCountry', '')}**",
        f"**{req.get('departDate', '—')} → {req.get('returnDate', '—')} · {req.get('travelers', req.get('passengers', 1))} traveler(s) · {nights} night(s) / {days} calendar day(s)**",
        f"Budget: **{req.get('budgetLevel', 'budget')}**",
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
                f"{'Non-stop' if stops == 0 else f'{stops} stop(s)'} |"
            )
        lines.append("\n*Live provider results; prices and availability can change.*")
    else:
        error = (
            services.get("flights", {}).get("error", "No live flight options were returned.")
            if isinstance(services.get("flights"), dict)
            else "No live flight options were returned."
        )
        lines.append(f"**Live flights unavailable.** {error}")

    lines += ["", "## 🏨 Hotels", ""]
    if hotels:
        lines += [
            "| Hotel | Nightly | Rating | Reviews |",
            "|---|---:|---:|---:|",
        ]
        for hotel in hotels[:6]:
            rating = hotel.get("rating", "—")
            if isinstance(rating, (int, float)):
                rating = f"{rating:.1f}"
            lines.append(
                f"| {hotel.get('name', 'Unknown')} | "
                f"{money(hotel.get('price'), hotel.get('currency', 'USD'))} | "
                f"{rating} | {hotel.get('reviews', '—')} |"
            )
        lines.append(
            f"\n*Live hotel rates returned for {req.get('departDate')} → {req.get('returnDate')}.*"
        )
    else:
        error = (
            services.get("hotels", {}).get("error", "No live hotel options were returned.")
            if isinstance(services.get("hotels"), dict)
            else "No live hotel options were returned."
        )
        lines.append(f"**Live hotels unavailable.** {error}")

    lines += ["", "## 📍 Things to do", ""]
    if attractions:
        for attraction in attractions[:8]:
            desc = attraction.get("description")
            lines.append(
                f"- **{attraction.get('name', 'Unnamed attraction')}**"
                + (f" — {desc}" if desc else "")
            )
    else:
        lines.append("- No high-quality live tourist attractions were returned.")

    lines += ["", "## 🍽️ Food picks", ""]
    if restaurants:
        for restaurant in restaurants[:8]:
            lines.append(f"- **{restaurant.get('name', 'Unnamed restaurant')}**")
        lines.append("\n*Recommendations only; no reservation is implied.*")
    else:
        lines.append("- No verified restaurant results were returned.")

    lines += ["", "## 🌦️ Weather", ""]
    weather_rows = weather.get("results", []) if isinstance(weather.get("results"), list) else []
    if weather_rows:
        lines += [
            "| Date | Temp | Feels like | Conditions | Humidity | Rain |",
            "|---|---:|---:|---|---:|---:|",
        ]
        for row in weather_rows:
            lines.append(
                f"| {row.get('date', '—')} | {row.get('temperature', '—')}°C | "
                f"{row.get('feelsLike', '—')}°C | {row.get('description', '—')} | "
                f"{row.get('humidity', '—')}% | {row.get('precipitationProbability', '—')}% |"
            )
        coverage = weather.get("coverage", {})
        if coverage:
            lines.append(
                f"\n*Live forecast coverage: **{coverage.get('returnedStart')} → "
                f"{coverage.get('returnedEnd')}**. No weather is extrapolated.*"
            )
    else:
        lines.append(
            f"**Live weather unavailable.** "
            f"{weather.get('error', 'No forecast data was returned for the requested dates.')}"
        )

    lines += ["", "## 💰 Budget", ""]
    if budget:
        currency = budget.get("currency", "USD")
        breakdown = budget.get("breakdown", {})
        lines.extend(
            [
                f"**Generic estimate:** {money(budget.get('total_budget'), currency)}",
                f"- Flights: {money(breakdown.get('flights_estimate'), currency)}",
                f"- Accommodation: {money(breakdown.get('accommodation_estimate'), currency)}",
                f"- Daily expenses: {money(breakdown.get('daily_expenses_estimate'), currency)}",
                "",
                "*Generic planning estimate only — not a live booking total.*",
            ]
        )

    if live.get("complete"):
        lines.extend(
            [
                "",
                f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'), live.get('currency', 'USD'))}",
                "",
                "Includes cheapest returned live flight + cheapest returned hotel nightly rate × nights; excludes food, local transport, activities, and unreturned taxes/fees.",
            ]
        )
    else:
        lines.append(
            "\n**Live-data subtotal:** incomplete because a usable live flight or hotel price was missing."
        )

    lines += ["", "## 🗓️ Suggested itinerary", ""]
    if start and end:
        cur = start
        attraction_index = 0
        restaurant_index = 0
        while cur <= end:
            day_no = (cur - start).days + 1
            lines.append(f"### Day {day_no} · {cur.isoformat()}")
            if day_no == 1:
                lines.append("- ✈️ Arrival / check-in")
            elif cur == end:
                lines.append("- 🧳 Check-out / departure")
            else:
                selected = []
                if attraction_index < len(attractions):
                    selected.append(attractions[attraction_index])
                    attraction_index += 1
                if (
                    attraction_index < len(attractions)
                    and len(selected) == 1
                    and day_no < days - 1
                    and len(attractions) >= 4
                ):
                    selected.append(attractions[attraction_index])
                    attraction_index += 1
                if selected:
                    for idx, attraction in enumerate(selected):
                        slot = "Morning" if idx == 0 else "Afternoon"
                        lines.append(f"- **{slot}:** {attraction.get('name')}")
                else:
                    lines.append("- Keep this period flexible rather than inventing another attraction.")

            if restaurants:
                restaurant = restaurants[restaurant_index % len(restaurants)]
                restaurant_index += 1
                lines.append(f"- 🍽️ **Food:** {restaurant.get('name')}")

            lines.append("")
            cur += timedelta(days=1)

    lines += [
        "## ⚠️ Notes",
        "",
        "- Live prices and availability can change before booking.",
        "- Generic budget estimates are not live booking totals.",
        "- Weather is shown only for dates actually covered by the live provider.",
        "- The itinerary uses only provider-returned attractions and restaurants.",
    ]
    return "\n".join(lines)


async def fetch_destinations_in_parallel(destinations: list[dict], status):
    async with AsyncExitStack() as stack:
        transport = await stack.enter_async_context(sse_client(server_url))
        session = await stack.enter_async_context(
            ClientSession(transport[0], transport[1])
        )

        async def fetch_one(item):
            city = item["destinationCity"]
            args = item["request"]
            try:
                trip = await get_trip(session, args)
                if trip.get("planningBlocked"):
                    return city, None, trip.get("error", "Trip planning blocked.")
                return city, trip, None
            except Exception as exc:
                return city, None, f"{type(exc).__name__}: {exc}"

        tasks = [asyncio.create_task(fetch_one(item)) for item in destinations]
        results = []
        for future in asyncio.as_completed(tasks):
            city, trip, error = await future
            results.append((city, trip, error))
            status.info(f"⚡ Live data received: {city}")
        return results


async def build_requests_from_route(route: dict, context: dict | None):
    base = dict((context or {}).get("request", {}))
    action = str(route.get("action", "ASK")).upper()

    if action == "BATCH_UPDATE":
        cities = [str(x).strip().title() for x in route.get("destinations", []) if str(x).strip()]
        airport_map = await resolve_unknown_airports(cities)
        requests = []
        for city in cities:
            req = normalize_request(
                {
                    "origin": base.get("origin"),
                    "destinationCity": city,
                    "destinationAirport": airport_map.get(city.lower(), ""),
                    "destinationCountry": COUNTRY.get(city.lower(), base.get("destinationCountry", "India")),
                    "departDate": base.get("departDate"),
                    "returnDate": base.get("returnDate"),
                    "passengers": base.get("passengers", base.get("travelers", 1)),
                    "budgetLevel": base.get("budgetLevel", "budget"),
                    "placesRadius": base.get("placesRadius", 5000),
                }
            )
            requests.append(req)
        return requests, action

    req = normalize_request(route, base)

    if action in ("UPDATE", "COMPARE"):
        req = normalize_request(
            {
                "origin": base.get("origin"),
                "destinationCity": route.get("destinationCity"),
                "destinationAirport": route.get("destinationAirport"),
                "destinationCountry": route.get("destinationCountry"),
                "departDate": base.get("departDate"),
                "returnDate": base.get("returnDate"),
                "passengers": base.get("passengers", base.get("travelers", 1)),
                "budgetLevel": base.get("budgetLevel", "budget"),
                "placesRadius": base.get("placesRadius", 5000),
            }
        )

    if not req.get("destinationAirport") and req.get("destinationCity"):
        airport_map = await resolve_unknown_airports([req["destinationCity"]])
        req["destinationAirport"] = airport_map.get(req["destinationCity"].lower(), "")

    return [req], action


async def run_turn(message: str, placeholder):
    async with AsyncExitStack():
        try:
            placeholder.info("⚡ Fast mode…")
            context = st.session_state.active_trip
            route = local_route(message, context)

            if route is None:
                placeholder.info("⚡ Fast router…")
                router = ChatGroq(
                    model="openai/gpt-oss-20b",
                    temperature=0,
                    max_tokens=180,
                )
                prompt = f"""
Return JSON only.

Current trip context:
{json.dumps((context or {}).get('request', {}), ensure_ascii=False)}

User message:
{message}

Allowed actions: PLAN, UPDATE, COMPARE, BATCH_UPDATE, REUSE, ASK.
- Preserve origin, dates, travelers and budget when the user says 'do the same'.
- 'do the same for A, B and C' means BATCH_UPDATE with exactly those destinations.
- Never invent dates.
- ALWAYS return airport fields as official uppercase 3-letter IATA codes.
- If a destination has no practical commercial passenger airport, use the nearest practical passenger airport.
- Never return a city name in an airport field.
- If an IATA code cannot be resolved confidently, leave it null.

Return:
{{"action":"...","origin":null,"destinationCity":null,
"destinationAirport":null,"destinations":[],"departDate":null,
"returnDate":null,"passengers":null,"budgetLevel":null}}
"""
                result = await router.ainvoke([HumanMessage(content=prompt)])
                raw = (result.content or "").strip()
                match = re.search(r"\{.*\}", raw, flags=re.S)
                if not match:
                    raise ValueError("Fast router returned invalid JSON")
                route = json.loads(match.group(0))

            action = str(route.get("action", "ASK")).upper()

            if action == "REUSE":
                if not st.session_state.active_trip:
                    placeholder.empty()
                    return "## 🧭 No active trip\n\nStart with a trip request first."
                low = message.lower()
                services = st.session_state.active_trip.get("services", {})
                if "cheapest hotel" in low:
                    hotels = [
                        h for h in service_list(services, "hotels")
                        if isinstance(h.get("price"), (int, float))
                    ]
                    if hotels:
                        hotel = min(hotels, key=lambda x: x["price"])
                        placeholder.empty()
                        return (
                            "### 🏨 Cheapest hotel\n\n"
                            f"**{hotel.get('name')}** — "
                            f"{money(hotel.get('price'), hotel.get('currency', 'USD'))}/night."
                        )
                if "cheapest flight" in low:
                    flights = [
                        f for f in service_list(services, "flights")
                        if isinstance(f.get("price"), (int, float))
                    ]
                    if flights:
                        flight = min(flights, key=lambda x: x["price"])
                        placeholder.empty()
                        return (
                            "### 🛫 Cheapest flight\n\n"
                            f"**{flight.get('airline')}** — "
                            f"{money(flight.get('price'), flight.get('currency', 'USD'))}."
                        )
                placeholder.empty()
                return render_trip(st.session_state.active_trip)

            if action == "ASK":
                placeholder.empty()
                return (
                    "## 🧭 I need a little more information\n\n"
                    "Please provide origin, destination, departure date, return date, and travelers."
                )

            requests, normalized_action = await build_requests_from_route(route, context)

            valid_requests = []
            errors = []
            for req in requests:
                ok, error = validate_request(req)
                if not ok:
                    errors.append(f"**{req.get('destinationCity', 'Trip')}:** {error}")
                else:
                    valid_requests.append(req)

            if errors and not valid_requests:
                placeholder.empty()
                return "## ⚠️ Cannot plan\n\n" + "\n\n".join(errors)

            if not valid_requests:
                placeholder.empty()
                return "## ⚠️ Cannot plan\n\n" + "\n\n".join(errors)

            destinations = [
                {"destinationCity": req["destinationCity"], "request": req}
                for req in valid_requests
            ]

            if normalized_action == "COMPARE" and not context:
                placeholder.empty()
                return (
                    "## 🧭 No active trip\n\n"
                    "Start with a trip first, then ask me to compare another destination."
                )

            if len(destinations) == 1:
                placeholder.info(f"⚡ Fetching live {destinations[0]['destinationCity']} data…")
            else:
                placeholder.info(
                    f"⚡ Fetching {len(destinations)} destinations in parallel…"
                )

            results = await fetch_destinations_in_parallel(destinations, placeholder)
            results.sort(key=lambda x: x[0].lower())

            rendered = []
            successful_trips = []
            for city, trip, error in results:
                if error:
                    rendered.append(f"## ❌ {city}\n\nLive trip data failed: `{error}`")
                    continue
                if not trip:
                    continue
                successful_trips.append(trip)
                st.session_state.stored_trips[city.lower()] = trip
                if normalized_action == "COMPARE":
                    st.session_state.comparison_trips[city.lower()] = trip
                else:
                    st.session_state.active_trip = trip
                rendered.append(
                    render_trip(
                        trip,
                        title_prefix=city if len(results) > 1 else "",
                    )
                )

            if errors:
                rendered.insert(0, "## ⚠️ Some requests were not run\n\n" + "\n\n".join(errors))

            if normalized_action == "COMPARE" and context and successful_trips:
                candidate = successful_trips[0]
                active_services = context.get("services", {})
                candidate_services = candidate.get("services", {})
                active_flights = service_list(active_services, "flights")
                candidate_flights = service_list(candidate_services, "flights")
                active_hotels = service_list(active_services, "hotels")
                candidate_hotels = service_list(candidate_services, "hotels")
                af = min((float(x["price"]) for x in active_flights if isinstance(x.get("price"), (int, float))), default=None)
                cf = min((float(x["price"]) for x in candidate_flights if isinstance(x.get("price"), (int, float))), default=None)
                ah = min((float(x["price"]) for x in active_hotels if isinstance(x.get("price"), (int, float))), default=None)
                ch = min((float(x["price"]) for x in candidate_hotels if isinstance(x.get("price"), (int, float))), default=None)
                an = int(context.get("request", {}).get("durationNights") or 0)
                cn = int(candidate.get("request", {}).get("durationNights") or 0)
                asub = af + ah * an if af is not None and ah is not None else None
                csub = cf + ch * cn if cf is not None and ch is not None else None
                rendered.append(
                    "## 🔎 Comparison summary\n\n"
                    "| Metric | Active trip | Candidate |\n"
                    "|---|---:|---:|\n"
                    f"| Cheapest flight | {money(af) if af is not None else '—'} | {money(cf) if cf is not None else '—'} |\n"
                    f"| Cheapest hotel/night | {money(ah) if ah is not None else '—'} | {money(ch) if ch is not None else '—'} |\n"
                    f"| Flight + hotel subtotal | {money(asub) if asub is not None else '—'} | {money(csub) if csub is not None else '—'} |"
                )

            placeholder.empty()
            return "\n\n---\n\n".join(rendered).strip()

        except Exception as exc:
            placeholder.empty()
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
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
