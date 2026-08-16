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
<span class="pill">● LIVE MCP · FAST MODE</span>
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
    st.success("🟢 Fast mode ready")
    st.caption("GPT-OSS 20B only · no NVIDIA")
    st.caption("Fresh travel facts always come from MCP")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_trip" not in st.session_state:
    st.session_state.active_trip = None
if "comparison_trips" not in st.session_state:
    st.session_state.comparison_trips = {}

IATA = {
    "chennai":"MAA","madras":"MAA","madurai":"IXM","coimbatore":"CJB","colombo":"CMB",
    "bangalore":"BLR","bengaluru":"BLR","hyderabad":"HYD","delhi":"DEL","new delhi":"DEL",
    "mumbai":"BOM","bombay":"BOM","kochi":"COK","ooty":"CJB","udhagamandalam":"CJB",
    "kodaikanal":"IXM","goa":"GOI","jaipur":"JAI","ahmedabad":"AMD","pune":"PNQ",
    "kolkata":"CCU","dubai":"DXB","singapore":"SIN","paris":"CDG","london":"LHR",
    "rome":"FCO","tokyo":"HND","new york":"JFK"
}
COUNTRY = {
    "chennai":"India","madurai":"India","coimbatore":"India","ooty":"India","udhagamandalam":"India",
    "kodaikanal":"India","goa":"India","jaipur":"India","ahmedabad":"India","pune":"India",
    "kolkata":"India","colombo":"Sri Lanka","dubai":"United Arab Emirates","singapore":"Singapore",
    "paris":"France","london":"United Kingdom","rome":"Italy","tokyo":"Japan","new york":"United States"
}

SYSTEM_REFERENCE = f"""
You are a lightweight travel-request resolver. Today's date is {date.today().isoformat()}.

MANDATORY RULES
- MCP is the source of truth for all fresh travel facts: flights, hotels, places, restaurants, weather, budget and currency.
- Never invent travel facts or prices.
- Preserve origin, dates, traveler count and budget from earlier turns.
- A new destination means fresh MCP data for that destination.
- ALWAYS use official uppercase 3-letter IATA codes for airport fields.
- Never put a city name in an airport field.
- Model knowledge may be used only to resolve ambiguous wording or an unknown airport, never to replace MCP live results.
- Comparison requests naming a new destination MUST fetch that candidate through MCP; never answer by replaying the active trip.

Return ONE compact JSON object only when the resolver is needed.
Allowed actions: PLAN, UPDATE, COMPARE, BATCH_UPDATE, REUSE, ASK.
Schema: {{"action":"ASK","origin":null,"destinationCity":null,"destinationAirport":null,"destinationCountry":null,"destinations":[],"departDate":null,"returnDate":null,"passengers":null,"budgetLevel":null}}
"""


def model():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=350)


def iso(v: Any):
    try:
        return datetime.strptime(str(v), "%Y-%m-%d").date()
    except Exception:
        return None


def money(v, currency="USD"):
    try:
        return f"{currency} {float(v):,.2f}"
    except Exception:
        return "Unavailable"


def normalize_city(v):
    return re.sub(r"\s+", " ", str(v or "").strip()).title()


def parse_dates(text):
    names = "January|February|March|April|May|June|July|August|September|October|November|December"
    found = re.findall(rf"\b(?:{names})\s+\d{{1,2}},?\s+\d{{4}}\b", text, re.I)
    out = []
    for raw in found[:2]:
        try:
            out.append(datetime.strptime(raw.replace(",", ""), "%B %d %Y").date().isoformat())
        except ValueError:
            pass
    return out if len(out) == 2 else re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)[:2]


def parse_nights(text):
    m = re.search(r"\b(\d+)\s*[- ]?night(?:s)?\b", text, re.I)
    return int(m.group(1)) if m else None


def split_destinations(text):
    text = re.sub(r"\s+(?:please|thanks)\s*$", "", text.strip(" .?"), flags=re.I)
    parts = re.split(r"\s*,\s*|\s+and\s+|\s*&\s*", text, flags=re.I)
    out, seen = [], set()
    for p in parts:
        p = normalize_city(p).strip(" .,-")
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def same_targets(text):
    patterns = [
        r"\bdo\s+the\s+same(?:\s+(?:trip|plan|planning|itinerary|thing))?\s+(?:with|for|in|to)\s+(.+)$",
        r"\bmake\s+the\s+same(?:\s+(?:trip|plan|planning|itinerary))?\s+(?:with|for|in|to)\s+(.+)$",
        r"\brepeat\s+(?:the\s+same\s+)?(?:trip|plan|planning|itinerary)\s+(?:for|in|with|to)\s+(.+)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text.strip(), re.I)
        if m:
            return split_destinations(m.group(1).rstrip("?"))
    return []


def comparison_target(text):
    """Extract a new destination from generic comparison phrasing."""
    cleaned = text.strip()

    patterns = [
        r"\bcompare\s+(?:this|that|the\s+(?:same|trip|plan|one|place|destination))\s+(?:with|to|for|vs|versus)\s+(.+)$",
        r"\bcompare\s+(?:it|this|that)\s+(?:with|to|vs|versus)\s+(.+)$",
        r"\bcompare\s+(?:with|to|for|vs|versus)\s+(.+)$",
        r"\bcompare\b.+?\b(?:with|to|for|vs|versus)\s+(.+)$",
    ]

    for pattern in patterns:
        m = re.search(pattern, cleaned, re.I)
        if not m:
            continue
        candidate = m.group(1).strip(" ?.")
        candidate = re.split(
            r"\s+(?:and|then)\s+(?:tell|say|let|show|give)\b|\s+(?:and\s+)?tell\s+me\b|\s+(?:and\s+)?which\b",
            candidate,
            maxsplit=1,
            flags=re.I,
        )[0]
        candidate = candidate.strip(" ,.-")
        if candidate:
            return normalize_city(candidate)

    return None


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


def local_route(message, context):
    text = message.strip()
    low = text.lower()
    base = dict((context or {}).get("request", {}))

    # Comparison MUST be checked before REUSE so a request such as
    # "compare the same for coimbatore" cannot be mistaken for reuse.
    candidate = comparison_target(text)
    if candidate:
        return {"action": "COMPARE", "destinationCity": candidate}

    if re.search(r"\bcheapest\s+(?:hotel|flight)\b", low):
        return {"action": "REUSE"}

    targets = same_targets(text)
    if targets:
        if len(targets) > 1:
            return {"action": "BATCH_UPDATE", "destinations": targets}
        return {"action": "UPDATE", "destinationCity": targets[0]}

    m = re.search(
        r"\b(?:change|switch|move)\s+(?:the\s+)?destination\s+(?:to|into)\s+([A-Za-z][\w\s.'-]*?)(?:\?|\.|$)",
        text,
        re.I,
    )
    if m:
        return {"action": "UPDATE", "destinationCity": normalize_city(m.group(1))}

    m = re.search(
        r"\bfrom\s+([A-Za-z][A-Za-z .'-]*?)\s+to\s+([A-Za-z][A-Za-z .'-]*?)(?=\s+(?:from|for|on|between|with)\b|\s*$)",
        text,
        re.I,
    )
    if m:
        base.update({
            "origin": normalize_city(m.group(1)),
            "destinationCity": normalize_city(m.group(2)),
        })

    dates = parse_dates(text)
    if len(dates) == 2:
        base["departDate"], base["returnDate"] = dates

    tm = re.search(r"\b(\d+)\s*(?:traveler|travellers|people|persons|adults?)\b", low)
    if tm:
        base["passengers"] = int(tm.group(1))

    n = parse_nights(text)
    if n is not None:
        base["requestedNights"] = n

    if any(w in low for w in ("budget", "cheap", "minimum")):
        base["budgetLevel"] = "budget"
    elif any(w in low for w in ("luxury", "luxurious")):
        base["budgetLevel"] = "luxury"
    elif "mid-range" in low or "moderate" in low:
        base["budgetLevel"] = "mid-range"

    if not base.get("destinationCity"):
        m = re.search(
            r"\b(?:to|for|in)\s+(?:the\s+)?(?:city\s+of\s+)?([A-Za-z][A-Za-z .'-]+?)(?:\s+from\s+|\s+for\s+\d|\s+on\s+|\s*$)",
            text,
            re.I,
        )
        if m:
            base["destinationCity"] = normalize_city(m.group(1))

    if base.get("origin") and base.get("destinationCity") and base.get("departDate") and base.get("returnDate"):
        return {"action": "PLAN", **base}

    return None


async def route_with_fallback(message, context):
    local = local_route(message, context)
    if local:
        return local

    prompt = f"""{SYSTEM_REFERENCE}
Current context: {json.dumps((context or {}).get('request', {}), ensure_ascii=False)}
User message: {message}

Preserve earlier trip context on follow-ups.
A new comparison destination MUST return COMPARE and MUST trigger fresh MCP data.
For 'do the same for A, B and C' return BATCH_UPDATE with destinations [A,B,C].
For questions answerable from saved live MCP data return REUSE.
If required data is missing return ASK.
"""

    try:
        result = await model().ainvoke([HumanMessage(content=prompt)])
        parsed = safe_json(result.content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {"action": "ASK"}


async def resolve_airports(cities):
    resolved = {}
    unknown = []

    for city in cities:
        key = normalize_city(city).lower()
        if key in IATA:
            resolved[key] = IATA[key]
        else:
            unknown.append(normalize_city(city))

    if not unknown:
        return resolved

    prompt = (
        f"{SYSTEM_REFERENCE}\nResolve practical commercial passenger airport IATA codes for: "
        f"{json.dumps(unknown, ensure_ascii=False)}. "
        "Return ONLY JSON mapping city to uppercase 3-letter IATA code; use null when no practical airport exists."
    )

    try:
        parsed = safe_json((await model().ainvoke([HumanMessage(content=prompt)])).content)
        if isinstance(parsed, dict):
            for city, code in parsed.items():
                if isinstance(code, str) and re.fullmatch(r"[A-Z]{3}", code.strip()):
                    resolved[str(city).lower()] = code.strip()
    except Exception:
        pass

    return resolved


def normalize_request(route, base=None):
    data = dict(base or {})
    for k, v in (route or {}).items():
        if v not in (None, ""):
            data[k] = v

    city = normalize_city(data.get("destinationCity")) if data.get("destinationCity") else ""
    origin_city = normalize_city(data.get("origin")) if data.get("origin") else ""

    data["origin"] = IATA.get(origin_city.lower(), origin_city.upper()) if origin_city else ""
    data["destinationCity"] = city
    data["destinationAirport"] = IATA.get(city.lower(), str(data.get("destinationAirport") or "").upper())
    data["destinationCountry"] = data.get("destinationCountry") or COUNTRY.get(city.lower(), "India")
    data["passengers"] = int(data.get("passengers") or data.get("travelers") or 1)
    data["budgetLevel"] = data.get("budgetLevel") or "budget"
    return data


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
    requested = req.get("requestedNights")
    if requested is not None and int(requested) != (end - start).days:
        return False, f"You specified {requested} night(s), but {req['departDate']} → {req['returnDate']} contains {(end-start).days} night(s)."
    if not re.fullmatch(r"[A-Z]{3}", req.get("origin", "")):
        return False, "Origin could not be normalized to an IATA airport code."
    return True, ""


async def mcp_trip(session, request):
    result = await asyncio.wait_for(session.call_tool("build_trip_data", arguments=request), timeout=20)
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
            c.startswith("tourism.") or "museum" in c or "culture" in c or "place_of_worship" in c or
            "historic" in c or "heritage" in c or c.startswith("natural") or "park" in c
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
        for f in flights[:5]:
            stops = int(f.get("stops", 0) or 0)
            out.append(f"| {f.get('airline','Unknown')} | {money(f.get('price'),f.get('currency','USD'))} | {f.get('departure','—')} | {f.get('arrival','—')} | {f.get('duration','—')} | {'Non-stop' if stops==0 else f'{stops} stop(s)'} |")
    else:
        error = services.get("flights", {}).get("error", "No live flight options were returned.") if isinstance(services.get("flights"), dict) else "No live flight options were returned."
        out.append(f"**Live flights unavailable.** {error}")

    out += ["", "## 🏨 Hotels", ""]
    if hotels:
        out += ["| Hotel | Nightly | Rating | Reviews |", "|---|---:|---:|---:|"]
        for h in hotels[:6]:
            out.append(f"| {h.get('name','Unknown')} | {money(h.get('price'),h.get('currency','USD'))} | {h.get('rating','—')} | {h.get('reviews','—')} |")
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
        for w in rows:
            out.append(f"| {w.get('date','—')} | {w.get('temperature','—')}°C | {w.get('feelsLike','—')}°C | {w.get('description','—')} | {w.get('precipitationProbability','—')}% |")
    else:
        out.append(f"**Live weather unavailable.** {weather.get('error','No forecast data was returned for the requested dates.')}")

    out += ["", "## 💰 Budget", ""]
    if budget:
        c = budget.get("currency", "USD")
        b = budget.get("breakdown", {})
        out += [
            f"**Generic estimate:** {money(budget.get('total_budget'),c)}",
            f"- Flights estimate: {money(b.get('flights_estimate'),c)}",
            f"- Accommodation estimate: {money(b.get('accommodation_estimate'),c)}",
            f"- Daily expenses estimate: {money(b.get('daily_expenses_estimate'),c)}",
        ]
    if live.get("complete"):
        out += ["", f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'),live.get('currency','USD'))}"]

    out += ["", "## 🗓️ Suggested itinerary", ""]
    if start and end:
        cur, ai, ri = start, 0, 0
        while cur <= end:
            d = (cur - start).days + 1
            out.append(f"### Day {d} · {cur.isoformat()}")
            if d == 1:
                out.append("- ✈️ Arrival / check-in")
            elif cur == end:
                out.append("- 🧳 Check-out / departure")
            else:
                picks = []
                if ai < len(atts):
                    picks.append(atts[ai]); ai += 1
                if ai < len(atts) and len(picks) == 1 and d < days - 1:
                    picks.append(atts[ai]); ai += 1
                if picks:
                    for j, a in enumerate(picks):
                        out.append(f"- **{'Morning' if j==0 else 'Afternoon'}:** {a.get('name')}")
                else:
                    out.append("- Keep this period flexible rather than inventing another attraction.")
            if rests:
                out.append(f"- 🍽️ **Food:** {rests[ri % len(rests)].get('name')}")
                ri += 1
            out.append("")
            cur += timedelta(days=1)

    out += [
        "## ⚠️ Notes", "",
        "- Live prices and availability can change before booking.",
        "- Fresh travel facts above were returned by the MCP service bundle.",
        "- No itinerary item is invented when the live provider returns nothing.",
    ]
    return "\n".join(out)


def compare(a, b):
    def low(trip, key):
        vals = [x.get("price") for x in listv(trip.get("services", {}), key) if isinstance(x.get("price"), (int, float))]
        return min(vals) if vals else None

    af, cf = low(a, "flights"), low(b, "flights")
    ah, ch = low(a, "hotels"), low(b, "hotels")
    an = int(a.get("request", {}).get("durationNights") or 0)
    cn = int(b.get("request", {}).get("durationNights") or 0)
    at = af + ah * an if af is not None and ah is not None else None
    ct = cf + ch * cn if cf is not None and ch is not None else None

    active_name = a.get("request", {}).get("destinationCity", "Active trip")
    candidate_name = b.get("request", {}).get("destinationCity", "Candidate")

    verdict = ""
    if at is not None and ct is not None:
        if at < ct:
            verdict = f"**Better for this budget-focused comparison: {active_name}.** Its cheapest live flight + hotel subtotal is lower."
        elif ct < at:
            verdict = f"**Better for this budget-focused comparison: {candidate_name}.** Its cheapest live flight + hotel subtotal is lower."
        else:
            verdict = "**Cost-wise, they are tied** on the cheapest live flight + hotel subtotal."
    else:
        verdict = "**I can't declare a cost winner from the returned live data alone** because one or more required prices were missing."

    return (
        "## 🔎 Comparison summary\n\n"
        "| Metric | Active trip | Candidate |\n|---|---:|---:|\n"
        f"| Cheapest flight | {money(af) if af is not None else '—'} | {money(cf) if cf is not None else '—'} |\n"
        f"| Cheapest hotel/night | {money(ah) if ah is not None else '—'} | {money(ch) if ch is not None else '—'} |\n"
        f"| Live flight + hotel subtotal | {money(at) if at is not None else '—'} | {money(ct) if ct is not None else '—'} |\n\n"
        f"**Destinations:** {active_name} vs {candidate_name}\n\n{verdict}"
    )


async def make_requests(route, context):
    base = dict((context or {}).get("request", {}))
    action = str(route.get("action", "ASK")).upper()

    if action == "BATCH_UPDATE":
        cities = [normalize_city(x) for x in route.get("destinations", []) if str(x).strip()]
        origin = base.get("origin", "")
        airport_lookup = await resolve_airports(cities + ([normalize_city(origin)] if origin else []))
        normalized_origin = airport_lookup.get(origin.lower(), origin.upper()) if origin else ""
        return [
            normalize_request({
                "origin": normalized_origin,
                "destinationCity": city,
                "destinationAirport": airport_lookup.get(city.lower(), ""),
                "destinationCountry": COUNTRY.get(city.lower(), "India"),
                "departDate": base.get("departDate"),
                "returnDate": base.get("returnDate"),
                "passengers": base.get("passengers", 1),
                "budgetLevel": base.get("budgetLevel", "budget"),
            })
            for city in cities
        ], action

    if action in ("UPDATE", "COMPARE"):
        city = normalize_city(route.get("destinationCity"))
        airport_lookup = await resolve_airports([city])
        return [
            normalize_request({
                "origin": base.get("origin"),
                "destinationCity": city,
                "destinationAirport": route.get("destinationAirport") or airport_lookup.get(city.lower(), ""),
                "destinationCountry": route.get("destinationCountry") or COUNTRY.get(city.lower(), "India"),
                "departDate": base.get("departDate"),
                "returnDate": base.get("returnDate"),
                "passengers": base.get("passengers", 1),
                "budgetLevel": base.get("budgetLevel", "budget"),
            })
        ], action

    return [normalize_request(route, base)], action


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

    results = await asyncio.gather(*[one(i, req) for i, req in enumerate(requests)])
    results.sort(key=lambda x: x[0])
    for index, trip, error in results:
        city = requests[index].get("destinationCity", "Trip")
        status.info(("⚠️ MCP failed" if error else "✅ MCP returned") + f": {city}")
    return results


async def run_turn(message, holder):
    stack = None
    try:
        holder.info("⚡ Fast mode…")
        context = st.session_state.active_trip
        route = await route_with_fallback(message, context)
        action = str(route.get("action", "ASK")).upper()

        if action == "REUSE":
            if not context:
                return "## 🧭 No active trip\n\nStart with a complete trip request first."
            services = context.get("services", {})
            low = message.lower()
            if "cheapest hotel" in low:
                hotels = [x for x in listv(services, "hotels") if isinstance(x.get("price"), (int, float))]
                if hotels:
                    h = min(hotels, key=lambda x: x["price"])
                    return f"### 🏨 Cheapest hotel\n\n**{h.get('name')}** — {money(h.get('price'),h.get('currency','USD'))}/night."
            if "cheapest flight" in low:
                flights = [x for x in listv(services, "flights") if isinstance(x.get("price"), (int, float))]
                if flights:
                    f = min(flights, key=lambda x: x["price"])
                    return f"### 🛫 Cheapest flight\n\n**{f.get('airline')}** — {money(f.get('price'),f.get('currency','USD'))}."
            return render_trip(context)

        if action == "ASK":
            return "## 🧭 I need a little more information\n\nPlease provide origin, destination, departure date, return date and travelers."

        if action == "COMPARE" and not context:
            return "## 🧭 No active trip\n\nStart with a trip first, then ask me to compare another destination."

        requests, effective = await make_requests(route, context)
        valid, errors = [], []
        for request in requests:
            ok, err = validate_request(request)
            if ok:
                valid.append(request)
            else:
                errors.append(f"**{request.get('destinationCity','Trip')}:** {err}")

        if not valid:
            return "## ⚠️ Cannot plan\n\n" + "\n\n".join(errors)

        holder.info(f"⚡ Fetching live MCP data for {len(valid)} destination(s)…")
        stack, session = await open_mcp()
        results = await fetch_parallel(session, valid, holder)

        rendered, successful = [], []
        for index, trip, error in results:
            city = valid[index].get("destinationCity", "Trip")
            if error:
                rendered.append(f"## ❌ {city}\n\nMCP live-data call failed: `{error}`")
                continue
            successful.append(trip)

        if effective == "COMPARE" and len(successful) == 1 and context:
            candidate = successful[0]
            candidate_name = candidate.get("request", {}).get("destinationCity", "candidate")
            st.session_state.comparison_trips[candidate_name.lower()] = candidate
            rendered.append(render_trip(candidate))
            rendered.append(compare(context, candidate))
            # IMPORTANT: comparison never replaces the active trip.
        elif effective == "UPDATE" and len(successful) == 1:
            st.session_state.active_trip = successful[0]
            rendered.append(render_trip(successful[0]))
        elif effective == "PLAN" and len(successful) == 1:
            st.session_state.active_trip = successful[0]
            rendered.append(render_trip(successful[0]))
        else:
            # BATCH_UPDATE fetches all new destinations through MCP in parallel,
            # but intentionally leaves the active trip unchanged.
            rendered.extend(render_trip(trip) for trip in successful)

        if errors:
            rendered.insert(0, "## ⚠️ Some requests were not run\n\n" + "\n\n".join(errors))

        return "\n\n---\n\n".join(rendered) if rendered else "## ⚠️ No live MCP results were returned."

    except Exception as exc:
        return f"## ❌ Something went wrong\n\n`{type(exc).__name__}: {exc}`"
    finally:
        if stack is not None:
            await stack.aclose()


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
