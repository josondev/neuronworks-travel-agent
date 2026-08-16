import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from datetime import date, datetime, timedelta
from typing import Any, List, Dict

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
:root{--bg:#080b14;--border:rgba(255,255,255,.10);--muted:#94a3b8}
.stApp{background:radial-gradient(circle at 10% 0%,rgba(37,99,235,.42),transparent 34%),radial-gradient(circle at 90% 10%,rgba(124,58,237,.36),transparent 32%),var(--bg)}
.block-container{max-width:1180px;padding-top:6rem;padding-bottom:6rem}
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
<p>Live flights · hotels · places · restaurants · weather · budget · currency</p>
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
    st.caption("Single model: openai/gpt-oss-20b on Groq")
    st.caption("No NVIDIA dependency")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_trip" not in st.session_state:
    st.session_state.active_trip = None
if "stored_trips" not in st.session_state:
    st.session_state.stored_trips = {}

IATA = {
    "chennai":"MAA", "madras":"MAA", "madurai":"IXM", "coimbatore":"CJB",
    "colombo":"CMB", "bangalore":"BLR", "bengaluru":"BLR", "hyderabad":"HYD",
    "delhi":"DEL", "new delhi":"DEL", "mumbai":"BOM", "bombay":"BOM", "kochi":"COK",
    "ooty":"CJB", "udhagamandalam":"CJB", "kodaikanal":"IXM"
}
COUNTRY = {
    "chennai":"India", "madurai":"India", "coimbatore":"India", "ooty":"India",
    "udhagamandalam":"India", "kodaikanal":"India", "colombo":"Sri Lanka"
}

TODAY = date.today().isoformat()

SYSTEM_REFERENCE = f"""
You are the fast routing brain of a factual travel agent.
Today's date: {TODAY}

ZERO-HALLUCINATION RULES
- Use only live MCP data for flights, hotels, places, restaurants, weather and budget.
- Never invent a flight, hotel, restaurant, attraction, weather value, availability or price.
- Report exact returned prices. Never lower a real price to fit a budget.
- Keep provider currency unless conversion is explicitly requested.
- If a service returns no results, report that honestly.

FOLLOW-UP / CONTEXT RULES
- Preserve origin, dates, traveler count and budget from earlier turns when the user says "do the same", "same trip", "repeat", or similar.
- A follow-up naming a new destination means FETCH FRESH DATA for that destination; it is not a request to reuse the old destination's live results.
- If several new destinations are named, fetch each destination independently and in parallel.
- Never discard the active trip just because comparison or batch destinations were requested.

IATA RULE — MANDATORY
- ALWAYS normalize airport values to official uppercase 3-letter IATA codes.
- Never send a city name to the flight search service.
- This applies to every destination, not only examples.
- If a destination has no practical commercial passenger airport, use the nearest practical airport while keeping the requested city separate.
- Never invent an IATA code. Resolve an unknown destination with the fast model before flight search.

BUDGET RULES
- budgetLevel must be exactly: budget, mid-range, or luxury.
- The budget tool is a generic planning estimate, not the live flight/hotel total.
- Separately report the live-data subtotal when a usable flight and hotel price exist.

DATE RULES
- Dates must be YYYY-MM-DD internally.
- Reject past dates.
- Return date must be after departure date.

MULTI-LEG (MULTI-HOP) RULES
- When the user asks for a trip with multiple stops (e.g., "Chennai → Madurai → Coimbatore"), return an action "MULTI_LEG" with a list of legs.
- Each leg is a dict with: origin, destination, departDate, returnDate, passengers, budgetLevel.
- The origin of the first leg is the overall starting point; the destination of the last leg is the final stop.
- For intermediate legs, the destination of leg i becomes the origin of leg i+1.
- Dates must be consecutive: the returnDate of leg i should be the same as or before the departDate of leg i+1 (if a same-day connection is desired, allow them to be equal).
- If the user gives a total duration but not per-leg dates, you may distribute the nights evenly, but prefer to ask for clarification if ambiguous.
"""

def model():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=500)

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

def service_list(services, key):
    value = services.get(key)
    return value if isinstance(value, list) else []

def parse_dates(text):
    names = "January|February|March|April|May|June|July|August|September|October|November|December"
    found = re.findall(rf"\b(?:{names})\s+\d{{1,2}},?\s+\d{{4}}\b", text, re.I)
    out = []
    for raw in found[:2]:
        try:
            out.append(datetime.strptime(raw.replace(",", ""), "%B %d %Y").date().isoformat())
        except ValueError:
            pass
    if len(out) == 2:
        return out
    return re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)[:2]

def parse_nights(text):
    m = re.search(r"\b(\d+)\s*[- ]?night(?:s)?\b", text, re.I)
    return int(m.group(1)) if m else None

def split_destinations(text):
    text = re.sub(r"\s+(?:please|thanks)\s*$", "", text.strip(" .?"), flags=re.I)
    parts = re.split(r"\s*,\s*|\s+and\s+|\s*&\s*", text, flags=re.I)
    seen, out = set(), []
    for part in parts:
        part = re.sub(r"^(?:the\s+)?(?:city\s+of\s+)", "", part.strip(), flags=re.I).strip(" .,-")
        if part and part.lower() not in seen:
            seen.add(part.lower())
            out.append(part.title())
    return out

def extract_same_targets(text):
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

def local_route(message, context):
    text = message.strip().lower()
    base = dict((context or {}).get("request", {}))

    if re.search(r"\bcheapest\s+(?:hotel|flight)\b", text):
        return {"action":"REUSE"}

    targets = extract_same_targets(text)
    if targets:
        if len(targets) > 1:
            return {"action":"BATCH_UPDATE", "destinations":targets}
        city = targets[0]
        return {"action":"UPDATE", "destinationCity":city, "destinationAirport":IATA.get(city.lower()), "destinationCountry":COUNTRY.get(city.lower())}

    m = re.search(r"\bcompare\b.*\b(?:with|vs|versus|to)\s+([a-zA-Z][\w\s.'-]*?)(?:\?|\.|$)", text)
    if m:
        city = m.group(1).strip().title()
        return {"action":"COMPARE", "destinationCity":city, "destinationAirport":IATA.get(city.lower()), "destinationCountry":COUNTRY.get(city.lower())}

    m = re.search(r"\b(?:change|switch|move)\s+(?:the\s+)?destination\s+(?:to|into)\s+([a-zA-Z][\w\s.'-]*?)(?:\?|\.|$)", text)
    if m:
        city = m.group(1).strip().title()
        return {"action":"UPDATE", "destinationCity":city, "destinationAirport":IATA.get(city.lower()), "destinationCountry":COUNTRY.get(city.lower())}

    m = re.search(r"\bfrom\s+([a-zA-Z][a-zA-Z .'-]*?)\s+to\s+([a-zA-Z][a-zA-Z .'-]*?)(?=\s+(?:from|for|on|between|with)\b|\s*$)", text)
    if m:
        origin, city = m.group(1).strip().title(), m.group(2).strip().title()
        base.update({"origin":IATA.get(origin.lower(), origin), "destinationCity":city, "destinationAirport":IATA.get(city.lower(), ""), "destinationCountry":COUNTRY.get(city.lower(), base.get("destinationCountry", "India"))})

    dates = parse_dates(text)
    if len(dates) == 2:
        base["departDate"], base["returnDate"] = dates
    m = re.search(r"\b(\d+)\s*(?:traveler|travellers|people|persons|adults?)\b", text)
    if m:
        base["passengers"] = int(m.group(1))
    elif re.search(r"\bfor\s+1\s+(?:traveler|person|adult)\b", text):
        base["passengers"] = 1
    nights = parse_nights(text)
    if nights is not None:
        base["requestedNights"] = nights
    if any(w in text for w in ("budget","cheap","minimum")):
        base["budgetLevel"] = "budget"
    elif any(w in text for w in ("luxury","luxurious")):
        base["budgetLevel"] = "luxury"
    elif any(w in text for w in ("mid-range","moderate")):
        base["budgetLevel"] = "mid-range"

    if not base.get("destinationCity"):
        m = re.search(r"\b(?:to|for|in)\s+(?:the\s+)?(?:city\s+of\s+)?([a-zA-Z][a-zA-Z .'-]+?)(?:\s+from\s+|\s+for\s+\d|\s+on\s+|\s*$)", text)
        if m:
            city = m.group(1).strip().title()
            base.update({"destinationCity":city, "destinationAirport":IATA.get(city.lower(), ""), "destinationCountry":COUNTRY.get(city.lower(), "India")})

    if base.get("origin") and base.get("destinationCity") and base.get("departDate") and base.get("returnDate"):
        return {"action":"PLAN", **base}
    return None

async def resolve_airports(cities):
    known = {city.lower(): IATA[city.lower()] for city in cities if city.lower() in IATA}
    unknown = [city for city in cities if city.lower() not in known]
    if not unknown:
        return known
    prompt = f"""{SYSTEM_REFERENCE}\n\nResolve practical commercial passenger airport IATA codes for these destinations:\n{json.dumps(unknown, ensure_ascii=False)}\nReturn ONLY a JSON object mapping each destination to an uppercase 3-letter IATA code."""
    try:
        result = await model().ainvoke([HumanMessage(content=prompt)])
        m = re.search(r"\{.*\}", (result.content or ""), re.S)
        if m:
            raw = json.loads(m.group(0))
            for city, code in raw.items():
                if isinstance(code, str) and re.fullmatch(r"[A-Z]{3}", code.strip()):
                    known[str(city).lower()] = code.strip()
    except Exception:
        pass
    return known

async def route_with_fallback(message, context):
    route = local_route(message, context)
    if route:
        return route
    prompt = f"""{SYSTEM_REFERENCE}\n\nReturn JSON only. Allowed actions: PLAN, UPDATE, COMPARE, BATCH_UPDATE, REUSE, ASK, MULTI_LEG.\nCurrent context:\n{json.dumps((context or {}).get('request', {}), ensure_ascii=False)}\nUser message:\n{message}\n\nRules:\n- 'do the same for A, B and C' => BATCH_UPDATE with exactly A, B, C.\n- Preserve earlier origin/dates/travelers/budget on follow-ups.\n- Never invent dates.\n- Always use uppercase 3-letter IATA codes in airport fields.\n- For multi-hop requests like "Chennai → Madurai → Coimbatore", return action "MULTI_LEG" with a "legs" list. Each leg has origin, destination, departDate, returnDate, passengers, budgetLevel.\n- Ensure dates are consecutive between legs.\n\nReturn: {{\"action\":\"...\",\"origin\":null,\"destinationCity\":null,\"destinationAirport\":null,\"destinations\":[],\"departDate\":null,\"returnDate\":null,\"passengers\":null,\"budgetLevel\":null,\"legs\":[]}}"""
    result = await model().ainvoke([HumanMessage(content=prompt)])
    m = re.search(r"\{.*\}", (result.content or ""), re.S)
    if not m:
        raise ValueError("Fast router returned invalid JSON")
    return json.loads(m.group(0))

def normalize(req, base=None):
    data = dict(base or {})
    for k, v in (req or {}).items():
        if v not in (None, ""):
            data[k] = v
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

def validate_request(req):
    start, end = iso(req.get("departDate")), iso(req.get("returnDate"))
    if not req.get("origin") or not req.get("destinationCity"):
        return False, "Please provide a valid origin and destination."
    if not start or not end:
        return False, "Please provide departure and return dates."
    if start < date.today():
        return False, f"The departure date {start} is in the past. Today is {date.today()}."
    if end <= start:
        return False, "Return date must be after departure date."
    requested = req.get("requestedNights")
    actual = (end - start).days
    if requested is not None and int(requested) != actual:
        return False, f"You specified {requested} night(s), but {req['departDate']} → {req['returnDate']} contains {actual} night(s). Please correct the dates or the night count."
    return True, ""

async def get_trip(session, args):
    result = await asyncio.wait_for(session.call_tool("build_trip_data", arguments=args), timeout=18)
    return json.loads(result.content[0].text)

def clean_attractions(items):
    tamil = "நகரம்|சாலை|தெரு|சந்து|குறுக்குச்சாலை|மாவட்டம்|மாநகராட்சி"
    deny = re.compile(rf"(\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|bus\s*station|railway|parking|signal|flyover|underpass|bypass|overpass|salai|theru|sandhu|mawatha|marg|nagar|colony|layout|township|extension|ward|sector|block|circle|chowk|hospital|water\s*works|car\s*shelter)\b)|({tamil})", re.I)
    cat_deny = ("administrative","populated_place","residential","postcode","suburb","neighbourhood","neighborhood","locality","commercial.building","office","hospital")
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict): continue
        name = str(item.get("name", "")).strip()
        if not name or deny.search(name): continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen: continue
        categories = [str(x).lower() for x in item.get("categories", [])]
        if any(any(bad in c for bad in cat_deny) for c in categories): continue
        allowed = any(c.startswith("tourism.") or "museum" in c or "culture" in c or "place_of_worship" in c or "historic" in c or "heritage" in c or c.startswith("natural") or "park" in c for c in categories)
        if not allowed: continue
        if re.search(r"\b(statue|viewpoint|train|triangle|building)\b", name, re.I) and not any(k in ",".join(categories) for k in ("historic","culture","museum","place_of_worship","heritage")): continue
        seen.add(key); out.append(item)
    return out

def clean_restaurants(items):
    deny = re.compile(r"\b(street|road|lane|mawatha|marg|salai|theru|sandhu|highway|junction|bus\s*stop|station|nagar|colony)\b", re.I)
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict): continue
        name = str(item.get("name", "")).strip()
        if not name or deny.search(name): continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen: continue
        categories = [str(x).lower() for x in item.get("categories", [])]
        if not any(c.startswith("catering.") for c in categories): continue
        seen.add(key); out.append(item)
    return out

def render_trip(trip, title_prefix=""):
    req, services, live = trip.get("request", {}), trip.get("services", {}), trip.get("liveDataSummary", {})
    flights = service_list(services, "flights")
    hotels = service_list(services, "hotels")
    attractions = clean_attractions(services.get("attractions"))
    restaurants = clean_restaurants(services.get("restaurants"))
    weather = services.get("weather") if isinstance(services.get("weather"), dict) else {}
    budget = services.get("budget") if isinstance(services.get("budget"), dict) else {}
    start, end = iso(req.get("departDate")), iso(req.get("returnDate"))
    nights = int(req.get("durationNights") or ((end - start).days if start and end else 0))
    days = nights + 1
    lines = [f"## ✈️ {title_prefix + ' ' if title_prefix else ''}Trip at a glance", "", f"**{req.get('origin','—')} → {req.get('destinationCity','—')}, {req.get('destinationCountry','')}**", f"**{req.get('departDate','—')} → {req.get('returnDate','—')} · {req.get('travelers', req.get('passengers',1))} traveler(s) · {nights} night(s) / {days} calendar day(s)**", f"Budget: **{req.get('budgetLevel','budget')}**", "", "## 🛫 Flights", ""]
    if flights:
        lines += ["| Airline | Price | Departure | Arrival | Duration | Stops |", "|---|---:|---|---|---:|---:|"]
        for f in flights[:5]:
            stops = int(f.get("stops",0) or 0)
            lines.append(f"| {f.get('airline','Unknown')} | {money(f.get('price'),f.get('currency','USD'))} | {f.get('departure','—')} | {f.get('arrival','—')} | {f.get('duration','—')} | {'Non-stop' if stops==0 else f'{stops} stop(s)'} |")
        lines.append("\n*Live provider results; prices and availability can change.*")
    else:
        error = services.get("flights",{}).get("error","No live flight options were returned.") if isinstance(services.get("flights"),dict) else "No live flight options were returned."
        lines.append(f"**Live flights unavailable.** {error}")

    lines += ["", "## 🏨 Hotels", ""]
    if hotels:
        lines += ["| Hotel | Nightly | Rating | Reviews |", "|---|---:|---:|---:|"]
        for h in hotels[:6]:
            rating = h.get("rating","—")
            if isinstance(rating,(int,float)): rating=f"{rating:.1f}"
            lines.append(f"| {h.get('name','Unknown')} | {money(h.get('price'),h.get('currency','USD'))} | {rating} | {h.get('reviews','—')} |")
        lines.append(f"\n*Live hotel rates returned for {req.get('departDate')} → {req.get('returnDate')}.*")
    else:
        error = services.get("hotels",{}).get("error","No live hotel options were returned.") if isinstance(services.get("hotels"),dict) else "No live hotel options were returned."
        lines.append(f"**Live hotels unavailable.** {error}")

    lines += ["", "## 📍 Things to do", ""]
    if attractions:
        lines.extend(f"- **{a.get('name','Unnamed attraction')}**" + (f" — {a.get('description')}" if a.get('description') else "") for a in attractions[:8])
    else:
        lines.append("- No high-quality live tourist attractions were returned. I won't pad the itinerary with roads, stations, wards, or random map objects.")

    lines += ["", "## 🍽️ Food picks", ""]
    if restaurants:
        lines.extend(f"- **{r.get('name','Unnamed restaurant')}**" for r in restaurants[:8])
        lines.append("\n*Recommendations only; no reservation is implied.*")
    else:
        lines.append("- No verified restaurant results were returned.")

    lines += ["", "## 🌦️ Weather", ""]
    weather_rows = weather.get("results", []) if isinstance(weather.get("results"), list) else []
    if weather_rows:
        lines += ["| Date | Temp | Feels like | Conditions | Humidity | Rain |", "|---|---:|---:|---|---:|---:|"]
        for w in weather_rows:
            lines.append(f"| {w.get('date','—')} | {w.get('temperature','—')}°C | {w.get('feelsLike','—')}°C | {w.get('description','—')} | {w.get('humidity','—')}% | {w.get('precipitationProbability','—')}% |")
        c = weather.get("coverage", {})
        if c: lines.append(f"\n*Live forecast coverage: **{c.get('returnedStart')} → {c.get('returnedEnd')}**. No weather is extrapolated.*")
    else:
        lines.append(f"**Live weather unavailable.** {weather.get('error','No forecast data was returned for the requested dates.')}")

    lines += ["", "## 💰 Budget", ""]
    if budget:
        cur = budget.get("currency","USD"); b = budget.get("breakdown",{})
        lines += [f"**Generic estimate:** {money(budget.get('total_budget'),cur)}", f"- Flights estimate: {money(b.get('flights_estimate'),cur)}", f"- Accommodation estimate: {money(b.get('accommodation_estimate'),cur)}", f"- Daily expenses estimate: {money(b.get('daily_expenses_estimate'),cur)}", "", "*Generic planning estimate only — not a live booking total.*"]
    if live.get("complete"):
        lines += ["", f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'), live.get('currency','USD'))}", "Includes cheapest returned live flight + cheapest returned hotel nightly rate × nights; excludes food, local transport, activities and unreturned taxes/fees."]
    else:
        lines.append("\n**Live-data subtotal:** incomplete because a usable live flight or hotel price was missing.")

    lines += ["", "## 🗓️ Suggested itinerary", ""]
    if start and end:
        cur = start; ai = 0; ri = 0
        while cur <= end:
            day_no = (cur-start).days + 1
            lines.append(f"### Day {day_no} · {cur.isoformat()}")
            if day_no == 1:
                lines.append("- ✈️ Arrival / check-in")
            elif cur == end:
                lines.append("- 🧳 Check-out / departure")
            else:
                picks = []
                if ai < len(attractions): picks.append(attractions[ai]); ai += 1
                if ai < len(attractions) and len(picks) == 1 and day_no < days-1 and len(attractions) >= 4:
                    picks.append(attractions[ai]); ai += 1
                if picks:
                    for j, a in enumerate(picks): lines.append(f"- **{'Morning' if j==0 else 'Afternoon'}:** {a.get('name')}")
                else:
                    lines.append("- Keep this period flexible rather than inventing another attraction.")
            if restaurants:
                lines.append(f"- 🍽️ **Food:** {restaurants[ri % len(restaurants)].get('name')}")
                ri += 1
            lines.append("")
            cur += timedelta(days=1)

    lines += ["## ⚠️ Notes", "", "- Live prices and availability can change before booking.", "- Generic budget estimates are not live booking totals.", "- Weather is shown only for dates covered by the live provider.", "- The itinerary uses only provider-returned attractions and restaurants."]
    return "\n".join(lines)

def render_multi_trip(trip_data):
    legs = trip_data.get("legs", [])
    if not legs:
        return "No trip data available."

    overall_lines = ["## ✈️ Multi‑City Trip at a glance", ""]
    first_req = legs[0].get("request", {})
    last_req = legs[-1].get("request", {})
    origin = first_req.get("origin", "—")
    total_travelers = first_req.get("passengers", 1)
    budget = first_req.get("budgetLevel", "budget")
    overall_lines.append(f"**{origin} → {' → '.join([leg['request']['destinationCity'] for leg in legs])}**")
    overall_lines.append(f"**{first_req.get('departDate','—')} → {last_req.get('returnDate','—')} · {total_travelers} traveler(s)**")
    overall_lines.append(f"Budget: **{budget}**")
    overall_lines.append("")

    leg_dates = []
    for leg in legs:
        req = leg.get("request", {})
        start = iso(req.get("departDate")); end = iso(req.get("returnDate"))
        leg_dates.append((start, end))

    overall_start = leg_dates[0][0]
    overall_end = leg_dates[-1][1]
    if not overall_start or not overall_end:
        overall_lines.append("⚠️ Date information missing for some legs.")
        return "\n".join(overall_lines)

    leg_flights = {}
    for i, leg in enumerate(legs):
        flights = service_list(leg.get("services", {}), "flights")
        if flights:
            leg_flights[i] = {"outbound": flights[0], "return": flights[-1] if len(flights) > 1 else None}
        else:
            leg_flights[i] = {"outbound": None, "return": None}

    overall_lines.append("## 🗓️ Combined Itinerary")
    overall_lines.append("")

    current_date = overall_start
    day_counter = 1
    while current_date <= overall_end:
        overall_lines.append(f"### Day {day_counter} · {current_date.isoformat()}")

        leg_idx = None
        for i, (start, end) in enumerate(leg_dates):
            if start <= current_date <= end:
                leg_idx = i
                break
        if leg_idx is None:
            overall_lines.append("- ⚠️ Date not covered by any leg.")
            current_date += timedelta(days=1); day_counter += 1
            continue

        leg = legs[leg_idx]
        req = leg.get("request", {})
        services = leg.get("services", {})
        attractions = clean_attractions(services.get("attractions"))
        restaurants = clean_restaurants(services.get("restaurants"))
        is_departure_day = current_date == leg_dates[leg_idx][0]
        is_last_leg_return_day = (leg_idx == len(legs)-1) and (current_date == leg_dates[leg_idx][1])

        if is_departure_day and leg_flights.get(leg_idx, {}).get("outbound"):
            flight = leg_flights[leg_idx]["outbound"]
            overall_lines.append(f"- ✈️ **Outbound flight:** {flight.get('airline','Unknown')} – {money(flight.get('price'), flight.get('currency','USD'))}")
        if is_last_leg_return_day and leg_flights.get(leg_idx, {}).get("return"):
            flight = leg_flights[leg_idx]["return"]
            overall_lines.append(f"- ✈️ **Return flight:** {flight.get('airline','Unknown')} – {money(flight.get('price'), flight.get('currency','USD'))}")

        if is_departure_day and leg_idx == 0:
            overall_lines.append("- 🏨 Check‑in at your hotel in " + req.get("destinationCity", "") + " (if applicable)")
        if is_last_leg_return_day:
            overall_lines.append("- 🧳 Check‑out and departure from " + req.get("destinationCity", ""))

        if attractions and not (is_departure_day and leg_idx == 0) and not is_last_leg_return_day:
            for attr in attractions[:2]:
                overall_lines.append(f"- **{attr.get('name')}**" + (f" — {attr.get('description')}" if attr.get('description') else ""))
        elif not is_departure_day and not is_last_leg_return_day:
            overall_lines.append("- Flexible day – no specific attraction scheduled.")

        if restaurants:
            overall_lines.append(f"- 🍽️ **Food:** {restaurants[0].get('name')} (example from {req.get('destinationCity','')})")
        else:
            overall_lines.append("- 🍽️ No restaurant recommendations available.")

        overall_lines.append("")
        current_date += timedelta(days=1); day_counter += 1

    overall_lines.append("## 📍 Leg‑by‑Leg Details")
    for i, leg in enumerate(legs):
        req = leg.get("request", {})
        overall_lines.append(f"### Leg {i+1}: {req.get('origin','?')} → {req.get('destinationCity','?')}")
        overall_lines.append(f"Dates: {req.get('departDate','—')} → {req.get('returnDate','—')}")
        overall_lines.append(f"Budget: {req.get('budgetLevel','budget')}")
        flights = service_list(leg.get("services",{}), "flights")
        if flights:
            numeric = [x for x in flights if isinstance(x.get('price'),(int,float))]
            if numeric:
                cheapest = min(numeric, key=lambda x: x['price'])
                overall_lines.append(f"- Cheapest flight: {money(cheapest.get('price'), cheapest.get('currency','USD'))}")
        hotels = service_list(leg.get("services",{}), "hotels")
        if hotels:
            numeric_h = [x for x in hotels if isinstance(x.get('price'),(int,float))]
            if numeric_h:
                cheapest_h = min(numeric_h, key=lambda x: x['price'])
                overall_lines.append(f"- Cheapest hotel/night: {money(cheapest_h.get('price'), cheapest_h.get('currency','USD'))}")
        overall_lines.append("")

    overall_lines.append("## ⚠️ Notes")
    overall_lines.append("- Live prices and availability can change before booking.")
    overall_lines.append("- Intermediate legs' return flights are not shown; only outbound flights for each leg and the final return flight are used.")
    overall_lines.append("- The combined itinerary is a suggestion based on live data from each stop.")
    return "\n".join(overall_lines)

async def fetch_many(requests, status):
    async with AsyncExitStack() as stack:
        transport = await stack.enter_async_context(sse_client(server_url))
        session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))

        async def one(req):
            city = req["destinationCity"]
            try:
                trip = await get_trip(session, req)
                if trip.get("planningBlocked"):
                    return city, None, trip.get("error", "Trip planning blocked.")
                return city, trip, None
            except Exception as exc:
                return city, None, f"{type(exc).__name__}: {exc}"

        tasks = [asyncio.create_task(one(req)) for req in requests]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            status.info(f"⚡ Live data received: {result[0]}")
        return results

async def build_requests(route, context):
    base = dict((context or {}).get("request", {}))
    action = str(route.get("action","ASK")).upper()

    if action == "MULTI_LEG":
        legs = route.get("legs", [])
        if not legs:
            raise ValueError("MULTI_LEG action requires a 'legs' list.")
        normalized_legs = []
        for leg in legs:
            req = normalize(leg, base)
            if not req.get("origin"):
                req["origin"] = base.get("origin", "")
            if not req.get("destinationCity"):
                req["destinationCity"] = leg.get("destination", "")
            if not req.get("destinationAirport") and req.get("destinationCity"):
                airports = await resolve_airports([req["destinationCity"]])
                req["destinationAirport"] = airports.get(req["destinationCity"].lower(), "")
            normalized_legs.append(req)
        return normalized_legs, action

    if action == "BATCH_UPDATE":
        cities = [str(x).strip().title() for x in route.get("destinations",[]) if str(x).strip()]
        airports = await resolve_airports(cities)
        return [normalize({
            "origin":base.get("origin"), "destinationCity":city,
            "destinationAirport":airports.get(city.lower(),""),
            "destinationCountry":COUNTRY.get(city.lower(), base.get("destinationCountry","India")),
            "departDate":base.get("departDate"), "returnDate":base.get("returnDate"),
            "passengers":base.get("passengers",1), "budgetLevel":base.get("budgetLevel","budget"),
        }) for city in cities], action

    req = normalize(route, base)
    if action in ("UPDATE","COMPARE"):
        airports = await resolve_airports([str(route.get("destinationCity") or "").strip()]) if route.get("destinationCity") else {}
        req = normalize({
            "origin":base.get("origin"), "destinationCity":route.get("destinationCity"),
            "destinationAirport":route.get("destinationAirport") or airports.get(str(route.get("destinationCity") or "").lower(),""),
            "destinationCountry":route.get("destinationCountry") or COUNTRY.get(str(route.get("destinationCity") or "").lower(),base.get("destinationCountry","India")),
            "departDate":base.get("departDate"), "returnDate":base.get("returnDate"),
            "passengers":base.get("passengers",1), "budgetLevel":base.get("budgetLevel","budget"),
        })
    if not req.get("destinationAirport") and req.get("destinationCity"):
        airports = await resolve_airports([req["destinationCity"]])
        req["destinationAirport"] = airports.get(req["destinationCity"].lower(),"")
    return [req], action

async def run_turn(message, placeholder):
    try:
        placeholder.info("⚡ Fast mode…")
        context = st.session_state.active_trip
        route = await route_with_fallback(message, context)
        action = str(route.get("action","ASK")).upper()

        if action == "REUSE":
            if not context:
                placeholder.empty(); return "## 🧭 No active trip\n\nStart with a complete trip request first."
            if isinstance(context, dict) and context.get("type") == "multi":
                placeholder.empty(); return render_multi_trip(context)
            services = context.get("services",{})
            low = message.lower()
            if "cheapest hotel" in low:
                hotels = [h for h in service_list(services,"hotels") if isinstance(h.get("price"),(int,float))]
                if hotels:
                    h = min(hotels,key=lambda x:x["price"])
                    placeholder.empty(); return f"### 🏨 Cheapest hotel\n\n**{h.get('name')}** — {money(h.get('price'),h.get('currency','USD'))}/night."
            if "cheapest flight" in low:
                flights = [f for f in service_list(services,"flights") if isinstance(f.get("price"),(int,float))]
                if flights:
                    f = min(flights,key=lambda x:x["price"])
                    placeholder.empty(); return f"### 🛫 Cheapest flight\n\n**{f.get('airline')}** — {money(f.get('price'),f.get('currency','USD'))}."
            placeholder.empty(); return render_trip(context)

        if action == "ASK":
            placeholder.empty(); return "## 🧭 I need a little more information\n\nPlease provide origin, destination, departure date, return date and travelers."

        if action == "COMPARE" and not context:
            placeholder.empty(); return "## 🧭 No active trip\n\nStart with a trip first, then ask me to compare another destination."

        requests, effective_action = await build_requests(route, context)

        valid_requests = []
        errors = []
        for req in requests:
            ok, err = validate_request(req)
            if ok: valid_requests.append(req)
            else: errors.append(f"**{req.get('destinationCity','Trip')}:** {err}")
        if not valid_requests:
            placeholder.empty(); return "## ⚠️ Cannot plan\n\n" + "\n\n".join(errors)

        if len(valid_requests) == 1:
            placeholder.info(f"⚡ Fetching live {valid_requests[0]['destinationCity']} data…")
        else:
            placeholder.info(f"⚡ Fetching {len(valid_requests)} destinations in parallel…")
        results = await fetch_many(valid_requests, placeholder)

        rendered, successful = [], []
        result_map = {city.lower(): (trip, error) for city, trip, error in results}
        for req in valid_requests:
            city = req["destinationCity"]
            trip, error = result_map.get(city.lower(), (None, "No result returned."))
            if error:
                rendered.append(f"## ❌ {city}\n\nLive trip data failed: `{error}`")
                continue
            successful.append(trip)
            st.session_state.stored_trips[city.lower()] = trip

        if effective_action == "MULTI_LEG" and successful:
            multi_trip = {"type": "multi", "legs": successful}
            st.session_state.active_trip = multi_trip
            rendered.append(render_multi_trip(multi_trip))
        elif effective_action in ("PLAN", "UPDATE") and len(successful) == 1:
            st.session_state.active_trip = successful[0]
            rendered.append(render_trip(successful[0]))
        else:
            for trip in successful:
                rendered.append(render_trip(trip))

        if errors:
            rendered.insert(0, "## ⚠️ Some requests were not run\n\n" + "\n\n".join(errors))

        if effective_action == "COMPARE" and context and successful:
            candidate = successful[0]
            if isinstance(context, dict) and context.get("type") == "multi":
                active_trip = context["legs"][0]
            else:
                active_trip = context
            a_services, c_services = active_trip.get("services",{}), candidate.get("services",{})
            af = min((float(x["price"]) for x in service_list(a_services,"flights") if isinstance(x.get("price"),(int,float))), default=None)
            cf = min((float(x["price"]) for x in service_list(c_services,"flights") if isinstance(x.get("price"),(int,float))), default=None)
            ah = min((float(x["price"]) for x in service_list(a_services,"hotels") if isinstance(x.get("price"),(int,float))), default=None)
            ch = min((float(x["price"]) for x in service_list(c_services,"hotels") if isinstance(x.get("price"),(int,float))), default=None)
            nights_a = int(active_trip.get("request",{}).get("durationNights") or 0)
            nights_c = int(candidate.get("request",{}).get("durationNights") or 0)
            sub_a = af + ah*nights_a if af is not None and ah is not None else None
            sub_c = cf + ch*nights_c if cf is not None and ch is not None else None
            active_name = active_trip.get("request",{}).get("destinationCity","Active trip")
            candidate_name = candidate.get("request",{}).get("destinationCity","Candidate")
            rendered.append("## 🔎 Comparison summary\n\n| Metric | Active trip | Candidate |\n|---|---:|---:|\n" + f"| Cheapest flight | {money(af) if af is not None else '—'} | {money(cf) if cf is not None else '—'} |\n| Cheapest hotel/night | {money(ah) if ah is not None else '—'} | {money(ch) if ch is not None else '—'} |\n| Live flight + hotel subtotal | {money(sub_a) if sub_a is not None else '—'} | {money(sub_c) if sub_c is not None else '—'} |\n\n**Destinations:** {active_name} vs {candidate_name}. Your active trip context remains unchanged.")

        placeholder.empty()
        return "\n\n---\n\n".join(rendered).strip()
    except Exception as exc:
        placeholder.empty()
        return f"## ❌ Something went wrong\n\n`{type(exc).__name__}: {exc}`"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Try: Chennai → Madurai → Coimbatore, Aug 20–28, 2 travelers"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = asyncio.run(run_turn(prompt, st.empty()))
        st.markdown(response)
        st.session_state.messages.append({"role":"assistant","content":response})
