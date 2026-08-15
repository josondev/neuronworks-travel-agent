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
<div class="hero">
<span class="pill">● LIVE MCP · FAST MODE</span>
<h1>✈️ Neuronworks Travel Agent</h1>
<p>Live flights · hotels · places · restaurants · weather · budget · currency</p>
</div>
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
    st.caption("Single fast model: openai/gpt-oss-20b on Groq")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "trip_context" not in st.session_state:
    st.session_state.trip_context = None

KNOWN_IATA = {
    "chennai":"MAA","madurai":"IXM","coimbatore":"CJB","colombo":"CMB",
    "bangalore":"BLR","bengaluru":"BLR","hyderabad":"HYD","delhi":"DEL",
    "mumbai":"BOM","kochi":"COK"
}


def parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Router did not return valid JSON")
    return json.loads(text[start:end+1])


def iso_date(value):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except Exception:
        return None


def as_list(value): return value if isinstance(value, list) else []
def as_dict(value): return value if isinstance(value, dict) else {}


def money(value, currency="USD"):
    try: return f"{currency} {float(value):,.2f}"
    except Exception: return "Unavailable"


def current_request():
    return (st.session_state.trip_context or {}).get("request", {})


def normalize_request(req: Dict[str, Any], base: Dict[str, Any] | None = None):
    base = base or {}
    city = str(req.get("destinationCity") or base.get("destinationCity") or "").strip()
    airport = str(req.get("destinationAirport") or base.get("destinationAirport") or "").strip().upper()
    if not airport and city:
        airport = KNOWN_IATA.get(city.lower(), "")
    return {
        "origin": str(req.get("origin") or base.get("origin") or "").strip().upper(),
        "destinationCity": city,
        "destinationCountry": str(req.get("destinationCountry") or base.get("destinationCountry") or "India").strip(),
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


async def route_request(user_message: str, context: Dict[str, Any] | None):
    base = context or {}
    base_request = base.get("request", {}) if isinstance(base, dict) else {}
    prompt = f"""
Return JSON only. You are the fast turn router for a travel agent.
Actions: PLAN, UPDATE, COMPARE, REUSE, ASK.

CURRENT TRIP CONTEXT (inherit any omitted fields):
{json.dumps(base_request, ensure_ascii=False)}

NEW USER MESSAGE:
{user_message}

Rules:
- Preserve context across turns.
- "compare this with X" => COMPARE, candidate destination X, inherit origin/dates/travelers/budget.
- "change destination to X" => UPDATE, inherit all other fields.
- A new trip missing origin or dates => ASK. Never invent dates.
- Follow-ups about cheapest/best/from existing results => REUSE.
- Infer well-known IATA codes when confident: Chennai MAA, Madurai IXM, Coimbatore CJB, Colombo CMB.

Return exactly:
{"action":"PLAN|UPDATE|COMPARE|REUSE|ASK","request":{"origin":null,"destinationCity":null,"destinationCountry":null,"destinationAirport":null,"departDate":null,"returnDate":null,"passengers":null,"budgetLevel":null,"currencyFrom":null,"currencyTo":null,"currencyAmount":1,"placesRadius":5000}}
"""
    model = ChatGroq(model="openai/gpt-oss-20b", temperature=0, max_tokens=220)
    result = await model.ainvoke([HumanMessage(content=prompt)])
    return parse_json(result.content or "")


async def call_bundle(session: ClientSession, args: Dict[str, Any]):
    result = await session.call_tool("build_trip_data", arguments=args)
    return json.loads(result.content[0].text)


def trip_valid(trip):
    req = trip.get("request", {})
    start, end = iso_date(req.get("departDate")), iso_date(req.get("returnDate"))
    today = date.today()
    if not start or not end:
        return False, "Please provide valid departure and return dates in YYYY-MM-DD format."
    if start < today:
        return False, f"The departure date {start.isoformat()} is in the past. Today is {today.isoformat()}."
    if end <= start:
        return False, "The return date must be after the departure date."
    return True, ""


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


def schedule(trip):
    req, services = trip.get("request", {}), trip.get("services", {})
    start, end = iso_date(req.get("departDate")), iso_date(req.get("returnDate"))
    attractions, restaurants = clean_attractions(services.get("attractions")), clean_restaurants(services.get("restaurants"))
    if not start or not end: return []
    dates=[]; cur=start
    while cur<=end: dates.append(cur); cur += timedelta(days=1)
    result=[]; ai=0; ri=0
    for i, d in enumerate(dates):
        picks=[] if i in (0,len(dates)-1) else attractions[ai:ai+2]
        ai += len(picks)
        restaurant=restaurants[ri] if restaurants else None
        if restaurants: ri=(ri+1)%len(restaurants)
        result.append({"date":d.isoformat(),"title":"Arrival" if i==0 else ("Departure" if i==len(dates)-1 else "Sightseeing"),"attractions":picks,"restaurant":restaurant})
    return result


def summary(trip):
    req, services = trip.get("request", {}), trip.get("services", {})
    flights, hotels = as_list(services.get("flights")), as_list(services.get("hotels"))
    attractions, restaurants = clean_attractions(services.get("attractions")), clean_restaurants(services.get("restaurants"))
    weather = as_dict(services.get("weather"))
    fp=min((float(x["price"]) for x in flights if isinstance(x.get("price"),(int,float))), default=None)
    hp=min((float(x["price"]) for x in hotels if isinstance(x.get("price"),(int,float))), default=None)
    nights=int(req.get("durationNights") or 0)
    return {"destination":req.get("destinationCity"),"flight":fp,"hotel":hp,"subtotal":(fp+hp*nights if fp is not None and hp is not None else None),"flights":len(flights),"hotels":len(hotels),"attractions":len(attractions),"restaurants":len(restaurants),"weatherDays":len(as_list(weather.get("results")))}


def render_trip(trip, title="## ✈️ Trip at a glance"):
    req, services, live = trip.get("request",{}), trip.get("services",{}), trip.get("liveDataSummary",{})
    flights, hotels = as_list(services.get("flights")), as_list(services.get("hotels"))
    attractions, restaurants = clean_attractions(services.get("attractions")), clean_restaurants(services.get("restaurants"))
    weather, budget = as_dict(services.get("weather")), as_dict(services.get("budget"))
    lines=[title,"",f"**{req.get('origin')} → {req.get('destinationCity')}, {req.get('destinationCountry')}**",f"**{req.get('departDate')} → {req.get('returnDate')} · {req.get('passengers',1)} traveler(s) · {req.get('durationNights')} night(s)**",f"Budget: **{req.get('budgetLevel','budget')}**",""]
    lines += ["## 🛫 Flights",""]
    if flights:
        lines += ["| Airline | Price | Departure | Arrival | Duration | Stops |","|---|---:|---|---|---:|---:|"]
        for f in flights[:5]:
            stops=int(f.get("stops",0) or 0); lines.append(f"| {f.get('airline','Unknown')} | {money(f.get('price'),f.get('currency','USD'))} | {f.get('departure','—')} | {f.get('arrival','—')} | {f.get('duration','—')} | {'Non-stop' if stops==0 else str(stops)+' stop(s)'} |")
    else: lines.append(f"**Live flights unavailable.** {as_dict(services.get('flights')).get('error','No live flight options returned.')}")
    lines += ["","## 🏨 Hotels",""]
    if hotels:
        lines += ["| Hotel | Nightly | Rating | Reviews |","|---|---:|---:|---:|"]
        for h in hotels[:6]:
            rating=f"{float(h['rating']):.1f}" if isinstance(h.get('rating'),(int,float)) else "—"; lines.append(f"| {h.get('name','Unknown')} | {money(h.get('price'),h.get('currency','USD'))} | {rating} | {h.get('reviews','—')} |")
    else: lines.append(f"**Live hotels unavailable.** {as_dict(services.get('hotels')).get('error','No live hotel options returned.')}")
    lines += ["","## 📍 Things to do",""]+[f"- **{p.get('name')}**" for p in attractions[:8]] or ["- No high-quality live tourist attractions were returned."]
    lines += ["","## 🍽️ Food picks",""]+[f"- **{r.get('name')}**" for r in restaurants[:8]] or ["- No verified restaurant results were returned."]
    lines += ["","## 🌦️ Weather",""]
    rows=as_list(weather.get("results"))
    if rows:
        lines += ["| Date | Temp | Feels like | Conditions | Rain |","|---|---:|---:|---|---:|"]
        for w in rows: lines.append(f"| {w.get('date','—')} | {w.get('temperature','—')}°C | {w.get('feelsLike','—')}°C | {w.get('description','—')} | {w.get('precipitationProbability','—')}% |")
    else: lines.append(f"**Live weather unavailable.** {weather.get('error','No forecast coverage returned.')}")
    lines += ["","## 💰 Budget",""]
    if budget:
        br=budget.get("breakdown",{}); cur=budget.get("currency","USD"); lines += [f"**Generic estimate:** {money(budget.get('total_budget'),cur)}",f"- Flights: {money(br.get('flights_estimate'),cur)}",f"- Accommodation: {money(br.get('accommodation_estimate'),cur)}",f"- Daily expenses: {money(br.get('daily_expenses_estimate'),cur)}"]
    if live.get("complete"): lines.append(f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'),live.get('currency','USD'))}")
    lines += ["","## 🗓️ Suggested itinerary",""]
    days=schedule(trip)
    for i,d in enumerate(days,1):
        lines.append(f"### Day {i} · {d['date']} · {d['title']}")
        if i==1: lines.append("- ✈️ Arrive and check in")
        for j,p in enumerate(d["attractions"]): lines.append(f"- **{'Morning' if j==0 else 'Afternoon'}:** {p.get('name')}")
        if d["restaurant"]: lines.append(f"- 🍽️ **Food:** {d['restaurant'].get('name')}")
        if i==len(days): lines.append("- 🧳 Check-out / departure")
        lines.append("")
    return "\n".join(lines)


def render_comparison(current, candidate):
    a,b=summary(current),summary(candidate); req=current.get("request",{})
    lines=[f"## 🔎 {a['destination']} vs {b['destination']}","",f"Same context: **{req.get('origin')} · {req.get('departDate')} → {req.get('returnDate')} · {req.get('passengers',1)} traveler(s) · {req.get('budgetLevel','budget')}**","","| Metric | Current | Candidate |","|---|---:|---:|"]
    for label,x,y in [("Cheapest flight",a["flight"],b["flight"]),("Cheapest hotel/night",a["hotel"],b["hotel"]),("Flight options",a["flights"],b["flights"]),("Hotel options",a["hotels"],b["hotels"]),("Verified attractions",a["attractions"],b["attractions"]),("Restaurants",a["restaurants"],b["restaurants"]),("Weather days",a["weatherDays"],b["weatherDays"])]:
        xv=money(x) if x is not None and "price" in label.lower() else ("—" if x is None else str(x)); yv=money(y) if y is not None and "price" in label.lower() else ("—" if y is None else str(y)); lines.append(f"| {label} | {xv} | {yv} |")
    lines += ["","### Recommendation",""]
    if a["subtotal"] is not None and b["subtotal"] is not None: lines.append(f"**Lower live flight + hotel subtotal:** {a['destination'] if a['subtotal']<b['subtotal'] else b['destination']}.")
    if a["attractions"]!=b["attractions"]: lines.append(f"**More verified attractions returned:** {a['destination'] if a['attractions']>b['attractions'] else b['destination']}.")
    lines.append("Your current trip context is preserved; this comparison does not replace the active trip.")
    return "\n".join(lines)


async def run_turn(user_message, placeholder):
    async with AsyncExitStack() as stack:
        placeholder.info("⚡ Fast router…")
        transport=await stack.enter_async_context(sse_client(server_url))
        session=await stack.enter_async_context(ClientSession(transport[0],transport[1]))
        action=await route_request(user_message,st.session_state.trip_context); act=str(action.get("action","ASK")).upper(); base=current_request(); req=normalize_request(action.get("request") or {},base)

        if act=="REUSE":
            if not st.session_state.trip_context: placeholder.empty(); return "## 🧭 No active trip yet\n\nStart with a complete trip request."
            lower=user_message.lower(); trip=st.session_state.trip_context
            if "cheapest hotel" in lower:
                hotels=as_list(trip.get("services",{}).get("hotels")); priced=[h for h in hotels if isinstance(h.get("price"),(int,float))]
                if priced:
                    h=min(priced,key=lambda x:x["price"]); placeholder.empty(); return f"### 🏨 Cheapest hotel\n\n**{h.get('name')}** — {money(h.get('price'),h.get('currency','USD'))}/night."
            if "cheapest flight" in lower:
                flights=as_list(trip.get("services",{}).get("flights")); priced=[f for f in flights if isinstance(f.get("price"),(int,float))]
                if priced:
                    f=min(priced,key=lambda x:x["price"]); placeholder.empty(); return f"### 🛫 Cheapest flight\n\n**{f.get('airline')}** — {money(f.get('price'),f.get('currency','USD'))}, {f.get('duration','—')}, {f.get('departure','—')} → {f.get('arrival','—')}."
            placeholder.empty(); return render_trip(trip)

        if act=="ASK":
            placeholder.empty(); return "## 🧭 I need a little more information\n\nPlease provide the origin, destination, departure date and return date. Existing trip context will be preserved on follow-ups."

        start,end=iso_date(req.get("departDate")),iso_date(req.get("returnDate")); today=date.today()
        if not req.get("destinationCity") or not start or not end:
            placeholder.empty(); return "## ⚠️ Missing trip details\n\nI could not resolve a destination or valid dates from this turn and the saved context."
        if start<today: placeholder.empty(); return f"## ⚠️ Past travel date\n\n{req['departDate']} is in the past. Today is {today.isoformat()}."
        if end<=start: placeholder.empty(); return "## ⚠️ Invalid dates\n\nReturn date must be after departure date."

        placeholder.info("⚡ Fetching live services in parallel…")
        candidate=await call_bundle(session,req)
        if candidate.get("planningBlocked"):
            placeholder.empty(); return f"## ⚠️ Trip cannot be planned\n\n{candidate.get('error','Unknown error')}"

        if act=="COMPARE" and st.session_state.trip_context:
            placeholder.empty(); return render_comparison(st.session_state.trip_context,candidate)

        st.session_state.trip_context=candidate
        placeholder.empty(); return render_trip(candidate)


for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if prompt:=st.chat_input("Try: Chennai → Madurai, Aug 20–25, 1 traveler"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try: response=asyncio.run(run_turn(prompt,st.empty()))
        except Exception as exc: response=f"## ❌ Something went wrong\n\n`{exc}`"
        st.markdown(response); st.session_state.messages.append({"role":"assistant","content":response})
