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
    "chennai":"MAA","madras":"MAA","madurai":"IXM","coimbatore":"CJB","colombo":"CMB",
    "bangalore":"BLR","bengaluru":"BLR","hyderabad":"HYD","delhi":"DEL","new delhi":"DEL",
    "mumbai":"BOM","bombay":"BOM","kochi":"COK"
}
COUNTRY = {"madurai":"India","coimbatore":"India","chennai":"India","colombo":"Sri Lanka"}


def iso(v):
    try:return datetime.strptime(str(v),"%Y-%m-%d").date()
    except:return None


def money(v,c="USD"):
    try:return f"{c} {float(v):,.2f}"
    except:return "Unavailable"


def llm_json(message, context):
    prompt=f'''Return JSON only. Preserve the existing trip context and interpret this turn.
Existing trip: {json.dumps(context or {}, ensure_ascii=False)}
User turn: {message}
Actions: PLAN, UPDATE, COMPARE, REUSE, ASK.
Examples:
- "compare this with Coimbatore" => COMPARE, destinationCity=Coimbatore
- "change destination to Coimbatore" => UPDATE, destinationCity=Coimbatore
- "which hotel is cheapest" => REUSE
- New trip missing dates/origin => ASK
Never invent dates. Infer these IATA codes only when obvious: Chennai MAA, Madurai IXM, Coimbatore CJB, Colombo CMB.
Return {{"action":"PLAN","destinationCity":null,"destinationCountry":null,"destinationAirport":null,"origin":null,"departDate":null,"returnDate":null,"passengers":null,"budgetLevel":null,"currencyFrom":null,"currencyTo":null,"currencyAmount":1}}'''
    model=ChatGroq(model="openai/gpt-oss-20b",temperature=0,max_tokens=180)
    result=asyncio.run(model.ainvoke([HumanMessage(content=prompt)]))
    txt=(result.content or "").strip()
    m=re.search(r"\{.*\}",txt,re.S)
    if not m: raise ValueError("Fast router returned invalid JSON")
    return json.loads(m.group(0))


def local_route(message, context):
    text=message.strip().lower()
    base=(context or {}).get("request",{}).copy()
    # deterministic follow-up handling: no LLM call for common cases
    compare=re.search(r"\bcompare\b.*\b(?:with|vs|versus|to)\s+([a-zA-Z][\w\s-]+?)(?:\?|\.|$)",text)
    change=re.search(r"\b(?:change|switch|move)\s+(?:the\s+)?destination\s+(?:to|into)\s+([a-zA-Z][\w\s-]+?)(?:\?|\.|$)",text)
    dest_phrase=re.search(r"\b(?:to|for)\s+(?:the\s+)?(?:city\s+of\s+)?(madurai|coimbatore|colombo|chennai)\b",text)
    if re.search(r"\bcheapest\s+(?:hotel|flight)\b",text): return {"action":"REUSE"}
    if compare:
        d=compare.group(1).strip().title(); d=re.sub(r"\s+$","",d)
        return {"action":"COMPARE","destinationCity":d,"destinationCountry":COUNTRY.get(d.lower()),"destinationAirport":IATA.get(d.lower())}
    if change:
        d=change.group(1).strip().title(); d=re.sub(r"\s+$","",d)
        return {"action":"UPDATE","destinationCity":d,"destinationCountry":COUNTRY.get(d.lower()),"destinationAirport":IATA.get(d.lower())}
    if dest_phrase:
        d=dest_phrase.group(1).title();
        if base.get("destinationCity") != d:
            base.update({"destinationCity":d,"destinationCountry":COUNTRY.get(d.lower()),"destinationAirport":IATA.get(d.lower())})
            return {"action":"PLAN",**base}
    # Parse explicit dates and traveler count from a new/updated request.
    dates=re.findall(r"\b20\d{2}-\d{2}-\d{2}\b",text)
    if len(dates)>=2:
        base["departDate"],base["returnDate"]=dates[0],dates[1]
    m=re.search(r"\b(\d+)\s*(?:traveler|travellers|people|persons|adult)",text)
    if m: base["passengers"]=int(m.group(1))
    if "budget" in text or "cheap" in text or "minimum" in text: base["budgetLevel"]="budget"
    if "luxury" in text: base["budgetLevel"]="luxury"
    if "mid-range" in text or "moderate" in text: base["budgetLevel"]="mid-range"
    if not base.get("destinationCity") and dest_phrase: base["destinationCity"]=dest_phrase.group(1).title()
    if base.get("destinationCity") and base.get("origin") and base.get("departDate") and base.get("returnDate"):
        return {"action":"PLAN",**base}
    # Use LLM only when the deterministic parser cannot resolve the turn.
    return None


def normalize(req, base):
    x=(base or {}).copy(); x.update({k:v for k,v in req.items() if v is not None and v!=""})
    city=str(x.get("destinationCity") or "").strip()
    origin=str(x.get("origin") or "").strip()
    # Always normalize city names to IATA codes for provider calls.
    x["origin"]=IATA.get(origin.lower(),origin.upper()) if origin else ""
    x["destinationAirport"]=x.get("destinationAirport") or IATA.get(city.lower(),"")
    x["destinationAirport"]=IATA.get(str(x["destinationAirport"]).lower(),str(x["destinationAirport"]).upper())
    x["destinationCountry"]=x.get("destinationCountry") or COUNTRY.get(city.lower(),"India")
    x["passengers"]=int(x.get("passengers") or 1)
    x["budgetLevel"]=x.get("budgetLevel") or "budget"
    x["placesRadius"]=int(x.get("placesRadius") or 5000)
    return x


def valid(req):
    start,end=iso(req.get("departDate")),iso(req.get("returnDate")); today=date.today()
    if not req.get("origin") or not req.get("destinationCity") or not req.get("destinationAirport"):
        return False,"Please provide origin and destination."
    if not start or not end:return False,"Please provide departure and return dates in YYYY-MM-DD format."
    if start<today:return False,f"The departure date {start} is in the past. Today is {today}."
    if end<=start:return False,"Return date must be after departure date."
    return True,""


async def get_trip(session,args):
    result=await asyncio.wait_for(session.call_tool("build_trip_data",arguments=args),timeout=17)
    return json.loads(result.content[0].text)


def clean_places(items,restaurants=False):
    bad=re.compile(r"\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|bus\s*station|railway|parking|signal|flyover|underpass|bypass|overpass|salai|theru|sandhu|mawatha|marg|nagar|colony|water\s*works|car\s*shelter|hospital)\b",re.I)
    out=[]; seen=set()
    for p in items if isinstance(items,list) else []:
        if not isinstance(p,dict):continue
        name=str(p.get("name","")).strip(); cats=','.join(str(c).lower() for c in p.get("categories",[]))
        if not name or bad.search(name):continue
        if restaurants and not any(c.startswith("catering.") for c in cats.split(',')):continue
        if not restaurants and not any(c.startswith("tourism.") or "museum" in c or "culture" in c or "place_of_worship" in c or "historic" in c or "heritage" in c or c.startswith("natural") or "park" in c for c in cats.split(',')):continue
        if not restaurants and re.search(r"\b(statue|viewpoint|train|triangle|building)\b",name,re.I) and not any(k in cats for k in ("historic","culture","museum","place_of_worship")):continue
        key=re.sub(r"[^a-z0-9]+"," ",name.lower()).strip()
        if key in seen:continue
        seen.add(key);out.append(p)
    return out


def stats(trip):
    s=trip.get("services",{}); req=trip.get("request",{})
    flights=s.get("flights",[]) if isinstance(s.get("flights"),list) else []
    hotels=s.get("hotels",[]) if isinstance(s.get("hotels"),list) else []
    a=clean_places(s.get("attractions")); r=clean_places(s.get("restaurants"),True)
    fp=min((float(x["price"]) for x in flights if isinstance(x.get("price"),(int,float))),default=None)
    hp=min((float(x["price"]) for x in hotels if isinstance(x.get("price"),(int,float))),default=None)
    nights=int(req.get("durationNights") or 0)
    return {"flight":fp,"hotel":hp,"subtotal":fp+hp*nights if fp is not None and hp is not None else None,"flights":len(flights),"hotels":len(hotels),"attractions":len(a),"restaurants":len(r)}


def comparison(a,b):
    x,y=stats(a),stats(b); ar=a["request"]; br=b["request"]
    lines=[f"## 🔎 {ar.get('destinationCity')} vs {br.get('destinationCity')}","",f"**Same trip context:** {ar.get('origin')} · {ar.get('departDate')} → {ar.get('returnDate')} · {ar.get('passengers',1)} traveler(s) · {ar.get('budgetLevel','budget')}","","| Metric | Madurai | Coimbatore |","|---|---:|---:|"]
    vals=[("Cheapest flight",x["flight"],y["flight"]),("Cheapest hotel/night",x["hotel"],y["hotel"]),("Flight options",x["flights"],y["flights"]),("Hotel options",x["hotels"],y["hotels"]),("Verified attractions",x["attractions"],y["attractions"]),("Restaurants",x["restaurants"],y["restaurants"])]
    for label,a1,b1 in vals:
        aa=money(a1) if "price" in label.lower() and a1 is not None else ("—" if a1 is None else str(a1));bb=money(b1) if "price" in label.lower() and b1 is not None else ("—" if b1 is None else str(b1));lines.append(f"| {label} | {aa} | {bb} |")
    lines += ["", "### Recommendation"]
    if x["subtotal"] is not None and y["subtotal"] is not None:
        lines.append(f"- Lower live flight + hotel subtotal: **{'Madurai' if x['subtotal']<y['subtotal'] else 'Coimbatore'}** ({money(min(x['subtotal'],y['subtotal']))}).")
    if x["attractions"]!=y["attractions"]:lines.append(f"- More verified attractions returned: **{'Madurai' if x['attractions']>y['attractions'] else 'Coimbatore'}**.")
    lines.append("- This comparison does **not** replace your active trip context.")
    return "\n".join(lines)


def render(trip):
    req,s=trip.get("request",{}),trip.get("services",{}); live=trip.get("liveDataSummary",{})
    flights=s.get("flights",[]) if isinstance(s.get("flights"),list) else []; hotels=s.get("hotels",[]) if isinstance(s.get("hotels"),list) else []
    attractions=clean_places(s.get("attractions"));restaurants=clean_places(s.get("restaurants"),True);weather=s.get("weather",{}) if isinstance(s.get("weather"),dict) else {};budget=s.get("budget",{}) if isinstance(s.get("budget"),dict) else {}
    lines=["## ✈️ Trip at a glance","",f"**{req.get('origin')} → {req.get('destinationCity')}, {req.get('destinationCountry')}**",f"**{req.get('departDate')} → {req.get('returnDate')} · {req.get('passengers',1)} traveler(s) · {req.get('durationNights')} night(s)**",f"Budget: **{req.get('budgetLevel','budget')}**","","## 🛫 Flights",""]
    if flights:
        lines += ["| Airline | Price | Departure | Arrival | Duration | Stops |","|---|---:|---|---|---:|---:|"]
        for f in flights[:5]:
            stp=int(f.get("stops",0) or 0);lines.append(f"| {f.get('airline','Unknown')} | {money(f.get('price'),f.get('currency','USD'))} | {f.get('departure','—')} | {f.get('arrival','—')} | {f.get('duration','—')} | {'Non-stop' if stp==0 else str(stp)+' stop(s)'} |")
    else:lines.append(f"**Live flights unavailable.** {s.get('flights',{}).get('error','No live flight results.') if isinstance(s.get('flights'),dict) else 'No live flight results.'}")
    lines += ["","## 🏨 Hotels",""]
    if hotels:
        lines += ["| Hotel | Nightly | Rating | Reviews |","|---|---:|---:|---:|"]
        for h in hotels[:6]: lines.append(f"| {h.get('name','Unknown')} | {money(h.get('price'),h.get('currency','USD'))} | {h.get('rating','—')} | {h.get('reviews','—')} |")
    else:lines.append("**Live hotels unavailable.**")
    lines += ["","## 📍 Things to do",""] + [f"- **{p.get('name')}**" for p in attractions[:8]]
    if not attractions:lines.append("- No high-quality live tourist attractions were returned.")
    lines += ["","## 🍽️ Food picks",""] + [f"- **{r.get('name')}**" for r in restaurants[:8]]
    if not restaurants:lines.append("- No verified restaurant results were returned.")
    lines += ["","## 🌦️ Weather",""]
    rows=weather.get("results",[]) if isinstance(weather.get("results"),list) else []
    if rows:
        for w in rows: lines.append(f"- **{w.get('date','—')}:** {w.get('temperature','—')}°C, feels like {w.get('feelsLike','—')}°C, {w.get('description','—')}, rain {w.get('precipitationProbability','—')}%")
    else:lines.append(f"- {weather.get('error','No live weather coverage returned.')}")
    lines += ["","## 💰 Budget",""]
    if budget:
        br=budget.get('breakdown',{});cur=budget.get('currency','USD');lines += [f"**Generic estimate:** {money(budget.get('total_budget'),cur)}",f"- Flights: {money(br.get('flights_estimate'),cur)}",f"- Accommodation: {money(br.get('accommodation_estimate'),cur)}",f"- Daily expenses: {money(br.get('daily_expenses_estimate'),cur)}"]
    if live.get('complete'):lines.append(f"\n**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'),live.get('currency','USD'))}")
    return "\n".join(lines)


async def run_turn(message, placeholder):
    async with AsyncExitStack() as stack:
        placeholder.info("⚡ Fast mode…")
        try:
            ctx=st.session_state.active_trip
            route=local_route(message,ctx)
            if route is None:
                route=llm_json(message,ctx)
            action=str(route.get("action","ASK")).upper()
            base=(ctx or {}).get("request",{})
            # Handle follow-up comparisons without destroying current context.
            if action=="COMPARE":
                candidate=normalize(route,base)
                ok,err=valid(candidate)
                if not ok:return f"## ⚠️ Cannot compare yet\n\n{err}"
                transport=await stack.enter_async_context(sse_client(server_url));session=await stack.enter_async_context(ClientSession(transport[0],transport[1]))
                placeholder.info("⚡ Fetching Coimbatore live data…")
                other=await get_trip(session,candidate)
                st.session_state.comparison_trips[candidate.get('destinationCity','Unknown')]=other
                placeholder.empty();return comparison(ctx,other)
            if action=="REUSE":
                if not ctx:return "## 🧭 No active trip\n\nStart with a complete trip request."
                low=message.lower();s=ctx.get('services',{})
                if "cheapest hotel" in low:
                    hs=s.get('hotels',[]) if isinstance(s.get('hotels'),list) else [];hs=[h for h in hs if isinstance(h.get('price'),(int,float))]
                    if hs:
                        h=min(hs,key=lambda z:z['price']);return f"### 🏨 Cheapest hotel\n\n**{h.get('name')}** — {money(h.get('price'),h.get('currency','USD'))}/night."
                if "cheapest flight" in low:
                    fs=s.get('flights',[]) if isinstance(s.get('flights'),list) else [];fs=[f for f in fs if isinstance(f.get('price'),(int,float))]
                    if fs:
                        f=min(fs,key=lambda z:z['price']);return f"### 🛫 Cheapest flight\n\n**{f.get('airline')}** — {money(f.get('price'),f.get('currency','USD'))}."
                return render(ctx)
            req=normalize(route,base)
            if not req.get('destinationCity') or not req.get('origin'):
                return "## 🧭 I need a little more information\n\nPlease provide the origin and destination."
            ok,err=valid(req)
            if not ok:return f"## ⚠️ {err}"
            transport=await stack.enter_async_context(sse_client(server_url));session=await stack.enter_async_context(ClientSession(transport[0],transport[1]))
            placeholder.info(f"⚡ Fetching live {req.get('destinationCity')} data…")
            trip=await get_trip(session,req)
            if trip.get('planningBlocked'):return f"## ⚠️ Trip cannot be planned\n\n{trip.get('error','Unknown error')}"
            st.session_state.active_trip=trip
            placeholder.empty();return render(trip)
        except Exception as exc:
            placeholder.empty()
            return f"## ❌ Something went wrong\n\n`{type(exc).__name__}: {exc}`"


for m in st.session_state.messages:
    with st.chat_message(m['role']):st.markdown(m['content'])

if prompt:=st.chat_input("Try: Chennai → Madurai, Aug 20–25, 1 traveler"):
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'):st.markdown(prompt)
    with st.chat_message('assistant'):
        response=asyncio.run(run_turn(prompt,st.empty()))
        st.markdown(response);st.session_state.messages.append({'role':'assistant','content':response})
