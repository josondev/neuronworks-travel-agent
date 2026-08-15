import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from datetime import date, datetime, timedelta
from typing import Any, Dict

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
.stApp{background:radial-gradient(circle at 10% 0%,rgba(37,99,235,.45) 0,transparent 34%),radial-gradient(circle at 90% 10%,rgba(124,58,237,.40) 0,transparent 32%),var(--bg)}
.block-container{max-width:1180px;padding-top:3rem;padding-bottom:6rem}
.hero{padding:26px 28px;border-radius:22px;margin-bottom:18px;background:linear-gradient(135deg,rgba(37,99,235,.26),rgba(124,58,237,.22));border:1px solid var(--border)}
.hero h1{margin:0;color:#fff;font-size:2.2rem;letter-spacing:-.04em}.hero p{margin:8px 0 0;color:#cbd5e1}
.pill{display:inline-block;padding:5px 11px;border-radius:999px;background:rgba(255,255,255,.09);color:#e2e8f0;font-size:.75rem;border:1px solid var(--border)}
section[data-testid="stSidebar"]{background:rgba(8,11,20,.96);border-right:1px solid var(--border)}
div[data-testid="stChatMessage"]{border:1px solid var(--border);border-radius:18px;padding:1rem 1.1rem;margin:.7rem 0;background:rgba(15,23,42,.82)}
div[data-testid="stChatMessageContent"]{color:var(--text)}
.muted{color:var(--muted);font-size:.82rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero"><span class="pill">● LIVE MCP · FAST MODE</span>
<h1>✈️ Neuronworks Travel Agent</h1>
<p>Live flights · hotels · places · restaurants · weather · budget · currency</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    server_url = st.text_input("MCP Server URL", value="https://neuronworks-travel-agent.onrender.com/sse")
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    if not groq_api_key:
        st.warning("Enter GROQ_API_KEY.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key
    st.success("🟢 Fast mode ready")
    st.caption("GPT-OSS 20B on Groq · local fast routing first")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "trip_context" not in st.session_state:
    st.session_state.trip_context = None

KNOWN_IATA = {
    "chennai": ("MAA", "India"), "madurai": ("IXM", "India"), "coimbatore": ("CJB", "India"),
    "colombo": ("CMB", "Sri Lanka"), "bangalore": ("BLR", "India"), "bengaluru": ("BLR", "India"),
    "hyderabad": ("HYD", "India"), "delhi": ("DEL", "India"), "mumbai": ("BOM", "India"),
    "kochi": ("COK", "India"), "trivandrum": ("TRV", "India"), "pondicherry": ("PNY", "India"),
}


def iso_date(v):
    try: return datetime.strptime(str(v), "%Y-%m-%d").date()
    except Exception: return None


def money(v, c="USD"):
    try: return f"{c} {float(v):,.2f}"
    except Exception: return "Unavailable"


def as_list(v): return v if isinstance(v, list) else []
def as_dict(v): return v if isinstance(v, dict) else {}


def normalize_request(req: Dict[str, Any], base: Dict[str, Any] | None = None):
    base = base or {}
    city = str(req.get("destinationCity") or base.get("destinationCity") or "").strip()
    airport = str(req.get("destinationAirport") or base.get("destinationAirport") or "").strip().upper()
    country = str(req.get("destinationCountry") or base.get("destinationCountry") or "").strip()
    if city.lower() in KNOWN_IATA:
        default_airport, default_country = KNOWN_IATA[city.lower()]
        airport = airport or default_airport
        country = country or default_country
    return {
        "origin": str(req.get("origin") or base.get("origin") or "").strip().upper(),
        "destinationCity": city,
        "destinationCountry": country or "India",
        "destinationAirport": airport,
        "departDate": str(req.get("departDate") or base.get("departDate") or "").strip(),
        "returnDate": str(req.get("returnDate") or base.get("returnDate") or "").strip(),
        "passengers": int(req.get("passengers") or base.get("passengers") or 1),
        "budgetLevel": str(req.get("budgetLevel") or base.get("budgetLevel") or "budget"),
        "currencyFrom": req.get("currencyFrom", base.get("currencyFrom")),
        "currencyTo": req.get("currencyTo", base.get("currencyTo")),
        "currencyAmount": req.get("currencyAmount", base.get("currencyAmount", 1)),
        "placesRadius": int(req.get("placesRadius") or base.get("placesRadius") or 5000),
    }


def parse_user_message_locally(message: str, context: Dict[str, Any] | None):
    """Fast, deterministic routing for the common travel turns.
    This avoids a network LLM call for the majority of requests and therefore
    avoids the unhandled TaskGroup failure that was happening in the Groq router.
    """
    text = message.strip()
    lower = text.lower()
    base = (context or {}).get("request", {}) if isinstance(context, dict) else {}

    # Existing-context follow-ups never need an LLM router.
    compare = re.search(r"(?:compare|comparison)\s+(?:this|that|it)\s+(?:with|to)\s+([A-Za-z][A-Za-z .'-]+?)(?:\.|$)", lower)
    if compare and context:
        city = compare.group(1).strip(" .").title()
        return "COMPARE", normalize_request({"destinationCity": city}, base)

    update = re.search(r"(?:change|switch|make)\s+(?:the\s+)?destination\s+(?:to|into)\s+([A-Za-z][A-Za-z .'-]+?)(?:\.|$)", lower)
    if update and context:
        city = update.group(1).strip(" .").title()
        return "UPDATE", normalize_request({"destinationCity": city}, base)

    if context and any(k in lower for k in ["cheapest hotel", "cheapest flight", "which hotel", "which flight", "best hotel", "best flight", "how much", "what is the budget"]):
        return "REUSE", normalize_request({}, base)

    # Explicitly stated city pair: from X to Y.
    pair = re.search(r"from\s+([A-Za-z][A-Za-z .'-]+?)\s+to\s+([A-Za-z][A-Za-z .'-]+?)(?:\s+from\s+|\s+on\s+|\s+for\s+|\.|$)", lower)
    origin = pair.group(1).strip().title() if pair else ""
    destination = pair.group(2).strip().title() if pair else ""
    if destination.lower() in KNOWN_IATA:
        destination_airport, destination_country = KNOWN_IATA[destination.lower()]
    else:
        destination_airport, destination_country = "", ""

    date_matches = re.findall(r"(\d{4}-\d{2}-\d{2})", text)
    if len(date_matches) >= 2:
        depart, ret = date_matches[0], date_matches[1]
    else:
        pretty = re.search(r"([A-Za-z]+\s+\d{1,2},\s*\d{4})\s*(?:to|[-–])\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", text)
        if pretty:
            try:
                depart = datetime.strptime(pretty.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
                ret = datetime.strptime(pretty.group(2), "%B %d, %Y").strftime("%Y-%m-%d")
            except ValueError:
                depart = ret = ""
        else:
            depart = ret = ""

    pax = None
    pax_match = re.search(r"(?:for\s+)?(\d+)\s*(?:traveler|travellers|travelers|people|persons|ppl)", lower)
    if pax_match: pax = int(pax_match.group(1))
    budget = "luxury" if "luxury" in lower else ("mid-range" if "mid-range" in lower or "moderate" in lower else ("budget" if "budget" in lower or "cheap" in lower or "minimum" in lower else None))

    if destination and (depart and ret):
        req = normalize_request({
            "origin": origin,
            "destinationCity": destination,
            "destinationCountry": destination_country or None,
            "destinationAirport": destination_airport,
            "departDate": depart,
            "returnDate": ret,
            "passengers": pax,
            "budgetLevel": budget,
        }, base if context else None)
        return ("UPDATE" if context and (origin or destination or depart or ret) else "PLAN"), req

    if context:
        return "REUSE", normalize_request({}, base)
    return None, None


async def llm_route_fallback(message: str, context: Dict[str, Any] | None):
    """Only used when deterministic routing cannot resolve the turn."""
    base = (context or {}).get("request", {}) if isinstance(context, dict) else {}
    prompt = f"""Return JSON only. Preserve omitted fields from CURRENT CONTEXT.
CURRENT CONTEXT: {json.dumps(base, ensure_ascii=False)}
USER: {message}
Choose one action: PLAN, UPDATE, COMPARE, REUSE, ASK.
Never invent dates. Infer only well-known IATA codes.
Schema: {{\"action\":\"PLAN|UPDATE|COMPARE|REUSE|ASK\",\"request\":{{\"origin\":null,\"destinationCity\":null,\"destinationCountry\":null,\"destinationAirport\":null,\"departDate\":null,\"returnDate\":null,\"passengers\":null,\"budgetLevel\":null}}}}"""
    model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=180, timeout=12)
    result = await model.ainvoke([HumanMessage(content=prompt)])
    raw = (result.content or "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start: raise ValueError("Router returned no JSON")
    data = json.loads(raw[start:end + 1])
    return str(data.get("action", "ASK")).upper(), normalize_request(data.get("request") or {}, base)


def clean_attractions(items):
    deny = re.compile(r"\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|bus\s*station|railway|parking|signal|flyover|underpass|bypass|overpass|salai|theru|sandhu|mawatha|marg|nagar|colony|layout|township|extension|ward|sector|block|circle|chowk|water\s*works|car\s*shelter|hospital)\b", re.I)
    out, seen = [], set()
    for item in as_list(items):
        if not isinstance(item, dict): continue
        name = str(item.get("name", "")).strip()
        if not name or deny.search(name): continue
        cats = [str(x).lower() for x in item.get("categories", [])]
        strong = any(c.startswith("tourism.") or "museum" in c or "culture" in c or "place_of_worship" in c or "historic" in c or "heritage" in c or c.startswith("natural") or "park" in c for c in cats)
        if not strong: continue
        if re.search(r"\b(statue|viewpoint|train|triangle|building)\b", name, re.I) and not any(x in ",".join(cats) for x in ("historic","culture","museum","place_of_worship")): continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen: continue
        seen.add(key); out.append(item)
    return out


def clean_restaurants(items):
    deny = re.compile(r"\b(street|road|lane|mawatha|marg|salai|theru|sandhu|highway|junction|bus\s*stop|station|nagar|colony)\b", re.I)
    out, seen = [], set()
    for item in as_list(items):
        if not isinstance(item, dict): continue
        name = str(item.get("name", "")).strip()
        if not name or deny.search(name): continue
        cats = [str(x).lower() for x in item.get("categories", [])]
        if not any(c.startswith("catering.") for c in cats): continue
        key = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
        if key in seen: continue
        seen.add(key); out.append(item)
    return out


def trip_valid(trip):
    req = trip.get("request", {})
    start, end = iso_date(req.get("departDate")), iso_date(req.get("returnDate"))
    today = date.today()
    if not start or not end: return False, "Please provide valid departure and return dates in YYYY-MM-DD format."
    if start < today: return False, f"The departure date {start.isoformat()} is in the past. Today is {today.isoformat()}."
    if end <= start: return False, "The return date must be after the departure date."
    return True, ""


def render_trip(trip):
    req, services, live = trip.get("request", {}), trip.get("services", {}), trip.get("liveDataSummary", {})
    flights, hotels = as_list(services.get("flights")), as_list(services.get("hotels"))
    attractions, restaurants = clean_attractions(services.get("attractions")), clean_restaurants(services.get("restaurants"))
    weather, budget = as_dict(services.get("weather")), as_dict(services.get("budget"))
    nights = int(req.get("durationNights") or 0)
    lines = [
        "## ✈️ Trip at a glance\n",
        f"**{req.get('origin')} → {req.get('destinationCity')}, {req.get('destinationCountry')}**  ",
        f"**{req.get('departDate')} → {req.get('returnDate')} · {req.get('passengers',1)} traveler(s) · {nights} night(s)**  ",
        f"Budget: **{req.get('budgetLevel','budget')}**\n",
        "## 🛫 Flights\n",
    ]
    if flights:
        lines += ["| Airline | Price | Departure | Arrival | Duration | Stops |", "|---|---:|---|---|---:|---:|"]
        for f in flights[:5]:
            stops=int(f.get("stops",0) or 0)
            lines.append(f"| {f.get('airline','Unknown')} | {money(f.get('price'),f.get('currency','USD'))} | {f.get('departure','—')} | {f.get('arrival','—')} | {f.get('duration','—')} | {'Non-stop' if stops==0 else str(stops)+' stop(s)'} |")
    else:
        lines.append(f"**Live flights unavailable.** {as_dict(services.get('flights')).get('error','No live flight options returned.')}")
    lines += ["", "## 🏨 Hotels\n"]
    if hotels:
        lines += ["| Hotel | Nightly | Rating | Reviews |", "|---|---:|---:|---:|"]
        for h in hotels[:6]:
            rating = f"{float(h['rating']):.1f}" if isinstance(h.get('rating'),(int,float)) else "—"
            lines.append(f"| {h.get('name','Unknown')} | {money(h.get('price'),h.get('currency','USD'))} | {rating} | {h.get('reviews','—')} |")
    else:
        lines.append(f"**Live hotels unavailable.** {as_dict(services.get('hotels')).get('error','No live hotel options returned.')}")
    lines += ["", "## 📍 Things to do\n"]
    if attractions:
        lines += [f"- **{p.get('name')}**" for p in attractions[:8]]
    else:
        lines.append("- No high-quality live tourist attractions were returned.")
    lines += ["", "## 🍽️ Food picks\n"]
    if restaurants:
        lines += [f"- **{r.get('name')}**" for r in restaurants[:8]]
    else:
        lines.append("- No verified restaurant results were returned.")
    lines += ["", "## 🌦️ Weather\n"]
    rows = as_list(weather.get("results"))
    if rows:
        lines += ["| Date | Temp | Feels like | Conditions | Rain |", "|---|---:|---:|---|---:|"]
        for w in rows:
            lines.append(f"| {w.get('date','—')} | {w.get('temperature','—')}°C | {w.get('feelsLike','—')}°C | {w.get('description','—')} | {w.get('precipitationProbability','—')}% |")
    else:
        lines.append(f"**Live weather unavailable.** {weather.get('error','No forecast coverage returned.')}")
    lines += ["", "## 💰 Budget\n"]
    if budget:
        br=budget.get("breakdown",{}); cur=budget.get("currency","USD")
        lines += [f"**Generic estimate:** {money(budget.get('total_budget'),cur)}", f"- Flights: {money(br.get('flights_estimate'),cur)}", f"- Accommodation: {money(br.get('accommodation_estimate'),cur)}", f"- Daily expenses: {money(br.get('daily_expenses_estimate'),cur)}"]
    if live.get("complete"):
        lines.append(f"\n**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'),live.get('currency','USD'))}")
    lines += ["", "## 🗓️ Suggested itinerary\n"]
    start, end = iso_date(req.get("departDate")), iso_date(req.get("returnDate"))
    dates=[]; cur=start
    while cur and end and cur<=end: dates.append(cur); cur += timedelta(days=1)
    ai=0
    for i,d in enumerate(dates):
        title = "Arrival" if i==0 else ("Departure" if i==len(dates)-1 else "Sightseeing")
        lines.append(f"### Day {i+1} · {d.isoformat()} · {title}")
        if i==0: lines.append("- ✈️ Arrive and check in")
        elif i==len(dates)-1: lines.append("- 🧳 Check-out / departure")
        else:
            picks=attractions[ai:ai+2]; ai+=len(picks)
            for j,p in enumerate(picks): lines.append(f"- **{'Morning' if j==0 else 'Afternoon'}:** {p.get('name')}")
            if restaurants: lines.append(f"- 🍽️ **Food:** {restaurants[(i-1)%len(restaurants)].get('name')}")
        lines.append("")
    lines += ["## ⚠️ Notes\n", "- Live prices and availability can change before booking.", "- Generic budget estimates are not live booking totals.", "- Weather is shown only for dates actually covered by the live provider.", "- Follow-up messages reuse the saved trip context."]
    return "\n".join(lines)


def trip_summary(trip):
    req, services = trip.get("request",{}), trip.get("services",{})
    flights, hotels = as_list(services.get("flights")), as_list(services.get("hotels"))
    attrs, restaurants = clean_attractions(services.get("attractions")), clean_restaurants(services.get("restaurants"))
    fp=min((float(x["price"]) for x in flights if isinstance(x.get("price"),(int,float))),default=None)
    hp=min((float(x["price"]) for x in hotels if isinstance(x.get("price"),(int,float))),default=None)
    nights=int(req.get("durationNights") or 0)
    return {"destination":req.get("destinationCity"),"flight":fp,"hotel":hp,"subtotal":fp+hp*nights if fp is not None and hp is not None else None,"flights":len(flights),"hotels":len(hotels),"attractions":len(attrs),"restaurants":len(restaurants)}


def render_comparison(current, candidate):
    a,b=trip_summary(current),trip_summary(candidate)
    req=current.get("request",{})
    lines=[f"## 🔎 {a['destination']} vs {b['destination']}", "", f"**Same dates:** {req.get('departDate')} → {req.get('returnDate')} · **{req.get('passengers',1)} traveler(s)** · **{req.get('budgetLevel','budget')}**", "", "| Metric | Current | Candidate |", "|---|---:|---:|"]
    rows=[("Cheapest flight",a["flight"],b["flight"]),("Cheapest hotel/night",a["hotel"],b["hotel"]),("Live flight options",a["flights"],b["flights"]),("Live hotel options",a["hotels"],b["hotels"]),("Verified attractions",a["attractions"],b["attractions"]),("Restaurants",a["restaurants"],b["restaurants"])]
    for label,x,y in rows:
        xv=money(x) if x is not None and "price" in label.lower() else ("—" if x is None else str(x)); yv=money(y) if y is not None and "price" in label.lower() else ("—" if y is None else str(y)); lines.append(f"| {label} | {xv} | {yv} |")
    if a["subtotal"] is not None and b["subtotal"] is not None:
        winner=a["destination"] if a["subtotal"]<b["subtotal"] else b["destination"]
        lines += ["", f"### 💡 Budget winner", f"**{winner}** has the lower live flight + hotel subtotal for this trip."]
    if a["attractions"] != b["attractions"]:
        winner=a["destination"] if a["attractions"]>b["attractions"] else b["destination"]
        lines.append(f"**{winner}** has more verified attractions returned by the live places service.")
    lines += ["", "_This comparison does not replace your active trip context._"]
    return "\n".join(lines)


async def call_bundle(session, args):
    result = await session.call_tool("build_trip_data", arguments=args)
    return json.loads(result.content[0].text)


async def run_turn(user_message, placeholder):
    # 1) Resolve the turn without a network call whenever possible.
    action, req = parse_user_message_locally(user_message, st.session_state.trip_context)
    if action is None:
        placeholder.info("⚡ Fast router…")
        try:
            action, req = await asyncio.wait_for(llm_route_fallback(user_message, st.session_state.trip_context), timeout=12)
        except Exception as exc:
            # Do not expose asyncio TaskGroup internals to the user. Preserve context
            # and ask for the one missing piece instead.
            placeholder.empty()
            if st.session_state.trip_context:
                return "## 🧭 I still have your active trip context.\n\nTell me what you want to change or compare (for example: **compare with Coimbatore**, **change destination to Coimbatore**, **cheapest hotel**)."
            return f"## 🧭 I couldn't parse that quickly.\n\nPlease provide origin, destination, departure date and return date."

    if action == "REUSE":
        trip = st.session_state.trip_context
        if not trip:
            placeholder.empty(); return "## 🧭 No active trip yet\n\nStart with a complete trip request."
        lower = user_message.lower()
        hotels = as_list(trip.get("services",{}).get("hotels")); flights = as_list(trip.get("services",{}).get("flights"))
        if "cheapest hotel" in lower and hotels:
            priced=[h for h in hotels if isinstance(h.get("price"),(int,float))]
            if priced:
                h=min(priced,key=lambda x:x["price"]); placeholder.empty(); return f"### 🏨 Cheapest hotel\n\n**{h.get('name')}** — {money(h.get('price'),h.get('currency','USD'))}/night."
        if "cheapest flight" in lower and flights:
            priced=[f for f in flights if isinstance(f.get("price"),(int,float))]
            if priced:
                f=min(priced,key=lambda x:x["price"]); placeholder.empty(); return f"### 🛫 Cheapest flight\n\n**{f.get('airline')}** — {money(f.get('price'),f.get('currency','USD'))}, {f.get('duration','—')}, {f.get('departure','—')} → {f.get('arrival','—')}."
        placeholder.empty(); return render_trip(trip)

    if action == "ASK" or not req:
        placeholder.empty(); return "## 🧭 I need a little more information\n\nPlease provide the origin, destination, departure date and return date. Your saved trip context will be preserved on follow-ups."

    start,end=iso_date(req.get("departDate")),iso_date(req.get("returnDate")); today=date.today()
    if not req.get("origin") or not req.get("destinationCity") or not start or not end:
        placeholder.empty(); return "## ⚠️ Missing trip details\n\nI need the origin, destination, departure date and return date. I will keep the rest of your saved context."
    if start<today:
        placeholder.empty(); return f"## ⚠️ Past travel date\n\n{req.get('departDate')} is in the past. Today is {today.isoformat()}."
    if end<=start:
        placeholder.empty(); return "## ⚠️ Invalid dates\n\nReturn date must be after departure date."

    placeholder.info("⚡ Fetching live services in parallel…")
    async with AsyncExitStack() as stack:
        transport=await stack.enter_async_context(sse_client(server_url))
        session=await stack.enter_async_context(ClientSession(transport[0],transport[1]))
        candidate=await asyncio.wait_for(call_bundle(session,req),timeout=16)

    if candidate.get("planningBlocked"):
        placeholder.empty(); return f"## ⚠️ Trip cannot be planned\n\n{candidate.get('error','Unknown error')}"

    if action=="COMPARE" and st.session_state.trip_context:
        placeholder.empty(); return render_comparison(st.session_state.trip_context,candidate)

    st.session_state.trip_context=candidate
    placeholder.empty(); return render_trip(candidate)


for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt:=st.chat_input("Try: Chennai → Madurai, Aug 20–25, 1 traveler"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            response=asyncio.run(run_turn(prompt,st.empty()))
        except Exception as exc:
            response="## ❌ Something went wrong\n\nThe live trip request failed. Please try the same request again."
        st.markdown(response)
        st.session_state.messages.append({"role":"assistant","content":response})
