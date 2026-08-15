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

IATA = {
    "chennai": "MAA", "madras": "MAA", "madurai": "IXM", "coimbatore": "CJB",
    "colombo": "CMB", "bangalore": "BLR", "bengaluru": "BLR", "hyderabad": "HYD",
    "delhi": "DEL", "new delhi": "DEL", "mumbai": "BOM", "bombay": "BOM", "kochi": "COK",
}
COUNTRY = {"madurai": "India", "coimbatore": "India", "chennai": "India", "colombo": "Sri Lanka"}


def iso(v):
    try:
        return datetime.strptime(str(v), "%Y-%m-%d").date()
    except Exception:
        return None


def money(v, c="USD"):
    try:
        return f"{c} {float(v):,.2f}"
    except Exception:
        return "Unavailable"


def parse_natural_dates(text):
    month_names = "January|February|March|April|May|June|July|August|September|October|November|December"
    pattern = rf"\b(?:{month_names})\s+\d{{1,2}},?\s+\d{{4}}\b"
    found = re.findall(pattern, text, flags=re.I)
    result = []
    for raw in found[:2]:
        for fmt in ("%B %d, %Y", "%B %d %Y"):
            try:
                result.append(datetime.strptime(raw.replace(",", ""), "%B %d %Y").date().isoformat())
                break
            except ValueError:
                continue
    if len(result) == 2:
        return result
    iso_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", text)
    return iso_dates[:2] if len(iso_dates) >= 2 else []


def normalize_request(req, base=None):
    x = dict(base or {})
    for k, v in (req or {}).items():
        if v not in (None, ""):
            x[k] = v

    city = str(x.get("destinationCity") or "").strip()
    origin = str(x.get("origin") or "").strip()
    airport = str(x.get("destinationAirport") or "").strip()

    x["origin"] = IATA.get(origin.lower(), origin.upper()) if origin else ""
    x["destinationAirport"] = IATA.get(airport.lower(), airport.upper()) if airport else IATA.get(city.lower(), "")
    x["destinationCountry"] = x.get("destinationCountry") or COUNTRY.get(city.lower(), "India")
    x["passengers"] = int(x.get("passengers") or 1)
    x["budgetLevel"] = x.get("budgetLevel") or "budget"
    x["placesRadius"] = int(x.get("placesRadius") or 5000)
    return x


def local_route(message, context):
    text = message.strip().lower()
    base = dict((context or {}).get("request", {}))

    if re.search(r"\bcheapest\s+(?:hotel|flight)\b", text):
        return {"action": "REUSE"}

    # "do the same with/for Coimbatore" means: preserve the active trip
    # context and change only the destination.
    same = re.search(r"\b(?:do|make)\s+(?:the\s+)?same\s+(?:trip\s+)?(?:with|for|in|to)\s+([a-zA-Z][\w\s-]*?)(?:\?|\.|$)", text)
    if same:
        city = same.group(1).strip().title()
        return {
            "action": "UPDATE",
            "destinationCity": city,
            "destinationAirport": IATA.get(city.lower()),
            "destinationCountry": COUNTRY.get(city.lower()),
        }

    compare = re.search(r"\bcompare\b.*\b(?:with|vs|versus|to)\s+([a-zA-Z][\w\s-]*?)(?:\?|\.|$)", text)
    if compare:
        city = compare.group(1).strip().title()
        return {"action": "COMPARE", "destinationCity": city, "destinationAirport": IATA.get(city.lower()), "destinationCountry": COUNTRY.get(city.lower())}

    change = re.search(r"\b(?:change|switch|move)\s+(?:the\s+)?destination\s+(?:to|into)\s+([a-zA-Z][\w\s-]*?)(?:\?|\.|$)", text)
    if change:
        city = change.group(1).strip().title()
        return {"action": "UPDATE", "destinationCity": city, "destinationAirport": IATA.get(city.lower()), "destinationCountry": COUNTRY.get(city.lower())}

    # Robustly parse: "from Chennai to Madurai ... from August 20, 2026 ..."
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

    if any(word in text for word in ("budget", "cheap", "minimum")):
        base["budgetLevel"] = "budget"
    elif any(word in text for word in ("luxury", "luxurious")):
        base["budgetLevel"] = "luxury"
    elif any(word in text for word in ("mid-range", "moderate")):
        base["budgetLevel"] = "mid-range"

    if not base.get("destinationCity"):
        dest = re.search(r"\b(?:to|for)\s+(?:the\s+)?(?:city\s+of\s+)?(madurai|coimbatore|colombo|chennai)\b", text)
        if dest:
            city = dest.group(1).title()
            base.update({"destinationCity": city, "destinationAirport": IATA.get(city.lower()), "destinationCountry": COUNTRY.get(city.lower())})

    if base.get("origin") and base.get("destinationCity") and base.get("departDate") and base.get("returnDate"):
        return {"action": "PLAN", **base}

    return None


async def llm_route(message, context):
    prompt = f"""Return JSON only. You are the fast travel-turn router.
Preserve context and fill only what the user changes.
CURRENT CONTEXT:
{json.dumps((context or {}).get('request', {}), ensure_ascii=False)}
USER TURN:
{message}

Actions: PLAN, UPDATE, COMPARE, REUSE, ASK.
- "do the same with/for X" => UPDATE, inherit origin/dates/travelers/budget and change only destination to X.
- compare this with X => COMPARE, inherit origin/dates/travelers/budget.
- change destination to X => UPDATE, inherit everything else.
- cheapest hotel/flight => REUSE.
- Missing required information on a new trip => ASK.
- Never invent dates.

IATA RULE — MANDATORY FOR EVERY PLACE:
- Always normalize origin and destination airport/city to the official uppercase 3-letter IATA code before returning JSON.
- Never return a city name in origin/destinationAirport when a standard IATA code exists.
- This applies to every current and future destination, not just examples.
- If you do not know an IATA code confidently, leave that field null rather than guessing.
- Examples: Chennai MAA, Madurai IXM, Coimbatore CJB, Colombo CMB, Bengaluru BLR, Mumbai BOM, Delhi DEL, Hyderabad HYD, Kochi COK.

Return exactly this JSON shape:
{{"action":"PLAN|UPDATE|COMPARE|REUSE|ASK","origin":null,"destinationCity":null,"destinationCountry":null,"destinationAirport":null,"departDate":null,"returnDate":null,"passengers":null,"budgetLevel":null,"currencyFrom":null,"currencyTo":null,"currencyAmount":1}}
"""
    model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=220)
    result = await model.ainvoke([HumanMessage(content=prompt)])
    text = (result.content or "").strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("Fast router returned invalid JSON")
    return json.loads(match.group(0))


def valid(req):
    start, end = iso(req.get("departDate")), iso(req.get("returnDate"))
    today = date.today()
    if not req.get("origin") or not req.get("destinationCity") or not req.get("destinationAirport"):
        return False, "Please provide a valid origin and destination."
    if not start or not end:
        return False, "Please provide departure and return dates in YYYY-MM-DD format."
    if start < today:
        return False, f"The departure date {start} is in the past. Today is {today}."
    if end <= start:
        return False, "Return date must be after departure date."
    return True, ""


async def get_trip(session, args):
    result = await asyncio.wait_for(session.call_tool("build_trip_data", arguments=args), timeout=17)
    return json.loads(result.content[0].text)


def clean_places(items, restaurants=False):
    bad = re.compile(r"\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|bus\s*station|railway|parking|signal|flyover|underpass|bypass|overpass|salai|theru|sandhu|mawatha|marg|nagar|colony|water\s*works|car\s*shelter|hospital)\b", re.I)
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        cats = ",".join(str(c).lower() for c in item.get("categories", []))
        if not name or bad.search(name):
            continue
        if restaurants and not any(c.startswith("catering.") for c in cats.split(",")):
            continue
        if not restaurants and not any(c.startswith("tourism.") or "museum" in c or "culture" in c or "place_of_worship" in c or "historic" in c or "heritage" in c or c.startswith("natural") or "park" in c for c in cats.split(",")):
            continue
        if not restaurants and re.search(r"\b(statue|viewpoint|train|triangle|building)\b", name, re.I) and not any(k in cats for k in ("historic", "culture", "museum", "place_of_worship")):
            continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def stats(trip):
    services = trip.get("services", {})
    req = trip.get("request", {})
    flights = services.get("flights", []) if isinstance(services.get("flights"), list) else []
    hotels = services.get("hotels", []) if isinstance(services.get("hotels"), list) else []
    attractions = clean_places(services.get("attractions"))
    restaurants = clean_places(services.get("restaurants"), True)
    cheapest_flight = min((float(x["price"]) for x in flights if isinstance(x.get("price"), (int, float))), default=None)
    cheapest_hotel = min((float(x["price"]) for x in hotels if isinstance(x.get("price"), (int, float))), default=None)
    nights = int(req.get("durationNights") or 0)
    subtotal = cheapest_flight + cheapest_hotel * nights if cheapest_flight is not None and cheapest_hotel is not None else None
    return {"flight": cheapest_flight, "hotel": cheapest_hotel, "subtotal": subtotal, "flights": len(flights), "hotels": len(hotels), "attractions": len(attractions), "restaurants": len(restaurants)}


def comparison(current, candidate):
    a, b = stats(current), stats(candidate)
    req = current.get("request", {})
    lines = [
        f"## 🔎 {current['request'].get('destinationCity')} vs {candidate['request'].get('destinationCity')}",
        "",
        f"**Same trip context:** {req.get('origin')} · {req.get('departDate')} → {req.get('returnDate')} · {req.get('passengers',1)} traveler(s) · {req.get('budgetLevel','budget')}",
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
    for label, x, y, is_money in rows:
        xv = money(x) if is_money and x is not None else ("—" if x is None else str(x))
        yv = money(y) if is_money and y is not None else ("—" if y is None else str(y))
        lines.append(f"| {label} | {xv} | {yv} |")
    lines += ["", "### Recommendation"]
    if a["subtotal"] is not None and b["subtotal"] is not None:
        winner = current['request'].get('destinationCity') if a['subtotal'] < b['subtotal'] else candidate['request'].get('destinationCity')
        lines.append(f"- Lower live flight + hotel subtotal: **{winner}** ({money(min(a['subtotal'], b['subtotal']))}).")
    if a["attractions"] != b["attractions"]:
        winner = current['request'].get('destinationCity') if a['attractions'] > b['attractions'] else candidate['request'].get('destinationCity')
        lines.append(f"- More verified attractions returned: **{winner}**.")
    lines.append("- This comparison does not replace your active trip context.")
    return "\n".join(lines)


def render(trip):
    req = trip.get("request", {})
    services = trip.get("services", {})
    live = trip.get("liveDataSummary", {})
    flights = services.get("flights", []) if isinstance(services.get("flights"), list) else []
    hotels = services.get("hotels", []) if isinstance(services.get("hotels"), list) else []
    attractions = clean_places(services.get("attractions"))
    restaurants = clean_places(services.get("restaurants"), True)
    weather = services.get("weather", {}) if isinstance(services.get("weather"), dict) else {}
    budget = services.get("budget", {}) if isinstance(services.get("budget"), dict) else {}

    lines = [
        "## ✈️ Trip at a glance", "",
        f"**{req.get('origin')} → {req.get('destinationCity')}, {req.get('destinationCountry')}**",
        f"**{req.get('departDate')} → {req.get('returnDate')} · {req.get('passengers',1)} traveler(s) · {req.get('durationNights')} night(s)**",
        f"Budget: **{req.get('budgetLevel','budget')}**", "",
        "## 🛫 Flights", "",
    ]
    if flights:
        lines += ["| Airline | Price | Departure | Arrival | Duration | Stops |", "|---|---:|---|---|---:|---:|"]
        for f in flights[:5]:
            stops = int(f.get("stops", 0) or 0)
            lines.append(f"| {f.get('airline','Unknown')} | {money(f.get('price'), f.get('currency','USD'))} | {f.get('departure','—')} | {f.get('arrival','—')} | {f.get('duration','—')} | {'Non-stop' if stops == 0 else str(stops)+' stop(s)'} |")
    else:
        flight_error = services.get("flights", {}).get("error", "No live flight results.") if isinstance(services.get("flights"), dict) else "No live flight results."
        lines.append(f"**Live flights unavailable.** {flight_error}")

    lines += ["", "## 🏨 Hotels", ""]
    if hotels:
        lines += ["| Hotel | Nightly | Rating | Reviews |", "|---|---:|---:|---:|"]
        for h in hotels[:6]:
            lines.append(f"| {h.get('name','Unknown')} | {money(h.get('price'),h.get('currency','USD'))} | {h.get('rating','—')} | {h.get('reviews','—')} |")
    else:
        lines.append("**Live hotels unavailable.**")

    lines += ["", "## 📍 Things to do", ""]
    lines.extend(f"- **{p.get('name')}**" for p in attractions[:8])
    if not attractions:
        lines.append("- No high-quality live tourist attractions were returned.")

    lines += ["", "## 🍽️ Food picks", ""]
    lines.extend(f"- **{r.get('name')}**" for r in restaurants[:8])
    if not restaurants:
        lines.append("- No verified restaurant results were returned.")

    lines += ["", "## 🌦️ Weather", ""]
    rows = weather.get("results", []) if isinstance(weather.get("results"), list) else []
    if rows:
        for w in rows:
            lines.append(f"- **{w.get('date','—')}:** {w.get('temperature','—')}°C, feels like {w.get('feelsLike','—')}°C, {w.get('description','—')}, rain {w.get('precipitationProbability','—')}%")
    else:
        lines.append(f"- {weather.get('error','No live weather coverage returned.')}")

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
        lines.append(f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'), live.get('currency','USD'))}")

    lines += ["", "## 🗓️ Suggested itinerary", ""]
    start, end = iso(req.get("departDate")), iso(req.get("returnDate"))
    if start and end:
        current = start
        ai = 0
        while current <= end:
            day_no = (current - start).days + 1
            lines.append(f"### Day {day_no} · {current.isoformat()}")
            if day_no == 1:
                lines.append("- ✈️ Arrive and check in")
            elif current == end:
                lines.append("- 🧳 Check-out / departure")
            else:
                for p in attractions[ai:ai+2]:
                    lines.append(f"- **Sightseeing:** {p.get('name')}")
                ai += min(2, max(0, len(attractions) - ai))
            lines.append("")
            current += timedelta(days=1)

    lines += ["## ⚠️ Notes", "", "- Live prices and availability can change before booking.", "- Generic budget estimates are not live booking totals.", "- Weather is shown only for dates actually covered by the live provider.", "- Follow-up messages reuse the saved trip context."]
    return "\n".join(lines)


async def run_turn(message, placeholder):
    async with AsyncExitStack() as stack:
        placeholder.info("⚡ Fast mode…")
        try:
            ctx = st.session_state.active_trip
            route = local_route(message, ctx)
            if route is None:
                placeholder.info("⚡ Fast router fallback…")
                route = await llm_route(message, ctx)

            action = str(route.get("action", "ASK")).upper()
            base = (ctx or {}).get("request", {})

            if action == "REUSE":
                if not ctx:
                    placeholder.empty()
                    return "## 🧭 No active trip\n\nStart with a complete trip request."
                lower = message.lower()
                services = ctx.get("services", {})
                if "cheapest hotel" in lower:
                    hotels = [h for h in services.get("hotels", []) if isinstance(h.get("price"), (int, float))]
                    if hotels:
                        h = min(hotels, key=lambda x: x["price"])
                        placeholder.empty()
                        return f"### 🏨 Cheapest hotel\n\n**{h.get('name')}** — {money(h.get('price'),h.get('currency','USD'))}/night."
                if "cheapest flight" in lower:
                    flights = [f for f in services.get("flights", []) if isinstance(f.get("price"), (int, float))]
                    if flights:
                        f = min(flights, key=lambda x: x["price"])
                        placeholder.empty()
                        return f"### 🛫 Cheapest flight\n\n**{f.get('airline')}** — {money(f.get('price'),f.get('currency','USD'))}."
                placeholder.empty()
                return render(ctx)

            req = normalize_request(route, base)
            if action == "COMPARE":
                if not ctx:
                    placeholder.empty()
                    return "## 🧭 No active trip\n\nStart with a complete trip first, then say 'compare this with Coimbatore'."
                ok, err = valid(req)
                if not ok:
                    placeholder.empty()
                    return f"## ⚠️ Cannot compare yet\n\n{err}"
                transport = await stack.enter_async_context(sse_client(server_url))
                session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
                placeholder.info(f"⚡ Fetching live {req.get('destinationCity')} data…")
                candidate = await get_trip(session, req)
                st.session_state.comparison_trips[req.get("destinationCity", "Unknown")] = candidate
                placeholder.empty()
                return comparison(ctx, candidate)

            if not req.get("destinationCity") or not req.get("origin"):
                placeholder.empty()
                return "## 🧭 I need a little more information\n\nPlease provide the origin and destination."

            ok, err = valid(req)
            if not ok:
                placeholder.empty()
                return f"## ⚠️ {err}"

            transport = await stack.enter_async_context(sse_client(server_url))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            placeholder.info(f"⚡ Fetching live {req.get('destinationCity')} data…")
            trip = await get_trip(session, req)
            if trip.get("planningBlocked"):
                placeholder.empty()
                return f"## ⚠️ Trip cannot be planned\n\n{trip.get('error','Unknown error')}"

            st.session_state.active_trip = trip
            placeholder.empty()
            return render(trip)
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
        st.session_state.messages.append({"role": "assistant", "content": response})
