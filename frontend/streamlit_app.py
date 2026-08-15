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

if "messages" not in st.session_state: st.session_state.messages=[]
if "active_trip" not in st.session_state: st.session_state.active_trip=None
if "comparison_trips" not in st.session_state: st.session_state.comparison_trips={}

IATA={
    "chennai":"MAA","madras":"MAA","madurai":"IXM","coimbatore":"CJB","colombo":"CMB",
    "bangalore":"BLR","bengaluru":"BLR","hyderabad":"HYD","delhi":"DEL","new delhi":"DEL",
    "mumbai":"BOM","bombay":"BOM","kochi":"COK"
}
COUNTRY={"madurai":"India","coimbatore":"India","chennai":"India","colombo":"Sri Lanka","ooty":"India"}


def iso(v):
    try:return datetime.strptime(str(v),"%Y-%m-%d").date()
    except Exception:return None


def money(v,c="USD"):
    try:return f"{c} {float(v):,.2f}"
    except Exception:return "Unavailable"


def parse_natural_dates(text):
    month_names="January|February|March|April|May|June|July|August|September|October|November|December"
    found=re.findall(rf"\b(?:{month_names})\s+\d{{1,2}},?\s+\d{{4}}\b",text,re.I)
    out=[]
    for raw in found[:2]:
        try: out.append(datetime.strptime(raw.replace(",",""),"%B %d %Y").date().isoformat())
        except ValueError: pass
    if len(out)==2:return out
    dates=re.findall(r"\b20\d{2}-\d{2}-\d{2}\b",text)
    return dates[:2] if len(dates)>=2 else []


def normalize_request(req,base=None):
    x=dict(base or {})
    for k,v in (req or {}).items():
        if v not in (None,""): x[k]=v
    city=str(x.get("destinationCity") or "").strip()
    origin=str(x.get("origin") or "").strip()
    airport=str(x.get("destinationAirport") or "").strip()
    x["origin"]=IATA.get(origin.lower(),origin.upper()) if origin else ""
    x["destinationAirport"]=IATA.get(airport.lower(),airport.upper()) if airport else IATA.get(city.lower(),"")
    x["destinationCountry"]=x.get("destinationCountry") or COUNTRY.get(city.lower(),"India")
    x["passengers"]=int(x.get("passengers") or 1)
    x["budgetLevel"]=x.get("budgetLevel") or "budget"
    x["placesRadius"]=int(x.get("placesRadius") or 5000)
    return x


def local_route(message,context,allow_unknown=False):
    text=message.strip().lower()
    base=dict((context or {}).get("request",{}))
    if re.search(r"\bcheapest\s+(?:hotel|flight)\b",text): return {"action":"REUSE"}

    # Generic same-trip follow-up: destination is the only changed field.
    same=re.search(r"\b(?:do|make)\s+(?:the\s+)?same(?:\s+(?:trip|plan|planning|itinerary))?\s+(?:with|for|in|to)\s+([a-zA-Z][\w\s.'-]*?)(?:\?|\.|$)",text)
    if same:
        city=same.group(1).strip().title()
        code=IATA.get(city.lower())
        if code or allow_unknown:
            return {"action":"UPDATE","destinationCity":city,"destinationAirport":code,"destinationCountry":COUNTRY.get(city.lower())}

    compare=re.search(r"\bcompare\b.*\b(?:with|vs|versus|to)\s+([a-zA-Z][\w\s.'-]*?)(?:\?|\.|$)",text)
    if compare:
        city=compare.group(1).strip().title()
        code=IATA.get(city.lower())
        if code or allow_unknown:
            return {"action":"COMPARE","destinationCity":city,"destinationAirport":code,"destinationCountry":COUNTRY.get(city.lower())}

    change=re.search(r"\b(?:change|switch|move)\s+(?:the\s+)?destination\s+(?:to|into)\s+([a-zA-Z][\w\s.'-]*?)(?:\?|\.|$)",text)
    if change:
        city=change.group(1).strip().title();code=IATA.get(city.lower())
        if code or allow_unknown:
            return {"action":"UPDATE","destinationCity":city,"destinationAirport":code,"destinationCountry":COUNTRY.get(city.lower())}

    route=re.search(r"\bfrom\s+([a-zA-Z][a-zA-Z .'-]*?)\s+to\s+([a-zA-Z][a-zA-Z .'-]*?)(?=\s+(?:from|for|on|between|with)\b|\s*$)",text)
    if route:
        origin=route.group(1).strip().title();city=route.group(2).strip().title()
        base.update({"origin":IATA.get(origin.lower(),origin),"destinationCity":city,"destinationAirport":IATA.get(city.lower(),""),"destinationCountry":COUNTRY.get(city.lower(),base.get("destinationCountry","India"))})

    dates=parse_natural_dates(text)
    if len(dates)==2: base["departDate"],base["returnDate"]=dates
    travelers=re.search(r"\b(\d+)\s*(?:traveler|travellers|people|persons|adults?)\b",text)
    if travelers: base["passengers"]=int(travelers.group(1))
    if any(w in text for w in ("budget","cheap","minimum")): base["budgetLevel"]="budget"
    elif any(w in text for w in ("luxury","luxurious")): base["budgetLevel"]="luxury"
    elif any(w in text for w in ("mid-range","moderate")): base["budgetLevel"]="mid-range"

    if not base.get("destinationCity"):
        dest=re.search(r"\b(?:to|for|in)\s+(?:the\s+)?(?:city\s+of\s+)?([a-zA-Z][a-zA-Z .'-]+?)(?:\s+from\s+|\s+between\s+|\s+for\s+\d|\s+on\s+|\s*$)",text)
        if dest:
            city=dest.group(1).strip().title();base.update({"destinationCity":city,"destinationAirport":IATA.get(city.lower(),""),"destinationCountry":COUNTRY.get(city.lower(),"India")})
    if base.get("origin") and base.get("destinationCity") and base.get("departDate") and base.get("returnDate"):
        return {"action":"PLAN",**base}
    return None


async def llm_route(message,context):
    prompt=f"""Return ONE line only in this exact format:
ACTION|ORIGIN_IATA|DESTINATION_CITY|DESTINATION_IATA|DEPART_DATE|RETURN_DATE|PASSENGERS|BUDGET

Current trip context: {json.dumps((context or {}).get('request',{}),ensure_ascii=False)}
User turn: {message}

Actions are PLAN, UPDATE, COMPARE, REUSE, ASK.
For 'do the same with/for X', preserve the current origin, dates, travelers and budget and change only X.
For 'compare with X', preserve the current trip and create X as a comparison candidate.
Dates must be YYYY-MM-DD and must never be invented.
IATA RULE: every airport value must be an official uppercase 3-letter IATA code. For destinations without their own commercial airport, choose the nearest practical commercial passenger airport and keep DESTINATION_CITY as the requested city. Never put a city name in an airport field.
If you cannot confidently resolve a required value, leave it empty after the pipe; do not fabricate.
"""
    model=ChatGroq(model="openai/gpt-oss-20b",temperature=0,max_tokens=80)
    result=await model.ainvoke([HumanMessage(content=prompt)])
    raw=(result.content or "").strip().splitlines()[0] if result.content else ""
    parts=[p.strip() for p in raw.split("|",7)]
    if len(parts)!=8: raise ValueError("Fast router returned invalid format")
    action,origin,city,airport,dep,ret,pax,budget=parts
    return {"action":action.upper(),"origin":origin or None,"destinationCity":city or None,"destinationAirport":airport or None,"departDate":dep or None,"returnDate":ret or None,"passengers":int(pax) if pax.isdigit() else None,"budgetLevel":budget or None}


def valid(req):
    start,end=iso(req.get("departDate")),iso(req.get("returnDate"));today=date.today()
    if not req.get("origin") or not req.get("destinationCity"):return False,"Please provide a valid origin and destination."
    if not start or not end:return False,"Please provide departure and return dates."
    if start<today:return False,f"The departure date {start} is in the past. Today is {today}."
    if end<=start:return False,"Return date must be after departure date."
    return True,""


async def get_trip(session,args):
    result=await asyncio.wait_for(session.call_tool("build_trip_data",arguments=args),timeout=17)
    return json.loads(result.content[0].text)


def clean_places(items,restaurants=False):
    bad=re.compile(r"\b(road|street|highway|lane|path|junction|roundabout|bus\s*stop|bus\s*station|railway|parking|signal|flyover|underpass|bypass|overpass|salai|theru|sandhu|mawatha|marg|nagar|colony|water\s*works|car\s*shelter|hospital|ward)\b",re.I)
    out=[];seen=set()
    for item in items if isinstance(items,list) else []:
        if not isinstance(item,dict):continue
        name=str(item.get("name","")).strip();cats=','.join(str(c).lower() for c in item.get("categories",[]))
        if not name or bad.search(name):continue
        if restaurants and not any(c.startswith("catering.") for c in cats.split(',')):continue
        if not restaurants and not any(c.startswith("tourism.") or "museum" in c or "culture" in c or "place_of_worship" in c or "historic" in c or "heritage" in c or c.startswith("natural") or "park" in c for c in cats.split(',')):continue
        if not restaurants and re.search(r"\b(statue|viewpoint|train|triangle|building)\b",name,re.I) and not any(k in cats for k in ("historic","culture","museum","place_of_worship")):continue
        key=re.sub(r"[^a-z0-9]+"," ",name.lower()).strip()
        if key in seen:continue
        seen.add(key);out.append(item)
    return out


def render(trip):
    req=trip.get("request",{});s=trip.get("services",{});live=trip.get("liveDataSummary",{})
    flights=s.get("flights",[]) if isinstance(s.get("flights"),list) else [];hotels=s.get("hotels",[]) if isinstance(s.get("hotels"),list) else []
    attractions=clean_places(s.get("attractions"));restaurants=clean_places(s.get("restaurants"),True);weather=s.get("weather",{}) if isinstance(s.get("weather"),dict) else {};budget=s.get("budget",{}) if isinstance(s.get("budget"),dict) else {}
    lines=["## ✈️ Trip at a glance","",f"**{req.get('origin')} → {req.get('destinationCity')}, {req.get('destinationCountry','')}**",f"**{req.get('departDate')} → {req.get('returnDate')} · {req.get('passengers',1)} traveler(s) · {req.get('durationNights')} night(s)**",f"Budget: **{req.get('budgetLevel','budget')}**","","## 🛫 Flights",""]
    if flights:
        lines += ["| Airline | Price | Departure | Arrival | Duration | Stops |","|---|---:|---|---|---:|---:|"]
        for f in flights[:5]:
            stops=int(f.get('stops',0) or 0);lines.append(f"| {f.get('airline','Unknown')} | {money(f.get('price'),f.get('currency','USD'))} | {f.get('departure','—')} | {f.get('arrival','—')} | {f.get('duration','—')} | {'Non-stop' if stops==0 else str(stops)+' stop(s)'} |")
    else: lines.append(f"**Live flights unavailable.** {s.get('flights',{}).get('error','No live flight results.') if isinstance(s.get('flights'),dict) else 'No live flight results.'}")
    lines += ["","## 🏨 Hotels",""]
    if hotels:
        lines += ["| Hotel | Nightly | Rating | Reviews |","|---|---:|---:|---:|"]
        for h in hotels[:6]:lines.append(f"| {h.get('name','Unknown')} | {money(h.get('price'),h.get('currency','USD'))} | {h.get('rating','—')} | {h.get('reviews','—')} |")
    else:lines.append("**Live hotels unavailable.**")
    lines += ["","## 📍 Things to do",""]+[f"- **{p.get('name')}**" for p in attractions[:8]]
    if not attractions:lines.append("- No high-quality live tourist attractions were returned.")
    lines += ["","## 🍽️ Food picks",""]+[f"- **{r.get('name')}**" for r in restaurants[:8]]
    if not restaurants:lines.append("- No verified restaurant results were returned.")
    lines += ["","## 🌦️ Weather",""]
    rows=weather.get('results',[]) if isinstance(weather.get('results'),list) else []
    if rows:
        for w in rows:lines.append(f"- **{w.get('date','—')}:** {w.get('temperature','—')}°C, feels like {w.get('feelsLike','—')}°C, {w.get('description','—')}, rain {w.get('precipitationProbability','—')}%")
    else:lines.append(f"- {weather.get('error','No live weather coverage returned.')}")
    lines += ["","## 💰 Budget",""]
    if budget:
        b=budget.get('breakdown',{});cur=budget.get('currency','USD');lines += [f"**Generic estimate:** {money(budget.get('total_budget'),cur)}",f"- Flights: {money(b.get('flights_estimate'),cur)}",f"- Accommodation: {money(b.get('accommodation_estimate'),cur)}",f"- Daily expenses: {money(b.get('daily_expenses_estimate'),cur)}"]
    if live.get('complete'):lines.append(f"**Cheapest live-data subtotal:** {money(live.get('cheapestLiveSubtotal'),live.get('currency','USD'))}")
    lines += ["","## 🗓️ Suggested itinerary",""]
    start,end=iso(req.get('departDate')),iso(req.get('returnDate'))
    if start and end:
        ai=0;cur=start
        while cur<=end:
            day=(cur-start).days+1;lines.append(f"### Day {day} · {cur.isoformat()}")
            if day==1:lines.append("- ✈️ Arrive and check in")
            elif cur==end:lines.append("- 🧳 Check-out / departure")
            else:
                for p in attractions[ai:ai+2]:lines.append(f"- **Sightseeing:** {p.get('name')}")
                ai=min(ai+2,len(attractions))
            lines.append("");cur+=timedelta(days=1)
    lines += ["## ⚠️ Notes","","- Live prices and availability can change before booking.","- Generic budget estimates are not live booking totals.","- Weather is shown only for dates actually covered by the live provider.","- Follow-up messages reuse the saved trip context."]
    return "\n".join(lines)


def compare_stats(a,b):
    def st(t):
        s=t.get('services',{});r=t.get('request',{});fs=s.get('flights',[]) if isinstance(s.get('flights'),list) else [];hs=s.get('hotels',[]) if isinstance(s.get('hotels'),list) else []
        fp=min((float(x['price']) for x in fs if isinstance(x.get('price'),(int,float))),default=None);hp=min((float(x['price']) for x in hs if isinstance(x.get('price'),(int,float))),default=None);n=int(r.get('durationNights') or 0)
        return fp,hp,(fp+hp*n if fp is not None and hp is not None else None),len(fs),len(hs),len(clean_places(s.get('attractions'))),len(clean_places(s.get('restaurants'),True))
    x,y=st(a),st(b);an=a['request'].get('destinationCity');bn=b['request'].get('destinationCity')
    lines=[f"## 🔎 {an} vs {bn}","",f"**Same trip context:** {a['request'].get('origin')} · {a['request'].get('departDate')} → {a['request'].get('returnDate')} · {a['request'].get('passengers',1)} traveler(s) · {a['request'].get('budgetLevel','budget')}","","| Metric | Current | Candidate |","|---|---:|---:|"]
    rows=[('Cheapest flight',x[0],y[0]),('Cheapest hotel/night',x[1],y[1]),('Flight options',x[3],y[3]),('Hotel options',x[4],y[4]),('Verified attractions',x[5],y[5]),('Restaurants',x[6],y[6])]
    for label,l,r in rows: lines.append(f"| {label} | {money(l) if 'price' in label.lower() and l is not None else ('—' if l is None else l)} | {money(r) if 'price' in label.lower() and r is not None else ('—' if r is None else r)} |")
    lines += ["","### Recommendation"]
    if x[2] is not None and y[2] is not None:lines.append(f"- Lower live flight + hotel subtotal: **{an if x[2]<y[2] else bn}** ({money(min(x[2],y[2]))}).")
    lines.append("- This comparison does not replace your active trip context.")
    return "\n".join(lines)


async def run_turn(message,placeholder):
    async with AsyncExitStack() as stack:
        placeholder.info("⚡ Fast mode…")
        try:
            ctx=st.session_state.active_trip
            route=local_route(message,ctx)
            if route is None: route=await llm_route(message,ctx)
            action=str(route.get('action','ASK')).upper();base=(ctx or {}).get('request',{})
            if action=='REUSE':
                if not ctx:return "## 🧭 No active trip\n\nStart with a complete trip request."
                low=message.lower();s=ctx.get('services',{})
                if 'cheapest hotel' in low:
                    hs=[h for h in s.get('hotels',[]) if isinstance(h.get('price'),(int,float))]
                    if hs:
                        h=min(hs,key=lambda z:z['price']);return f"### 🏨 Cheapest hotel\n\n**{h.get('name')}** — {money(h.get('price'),h.get('currency','USD'))}/night."
                if 'cheapest flight' in low:
                    fs=[f for f in s.get('flights',[]) if isinstance(f.get('price'),(int,float))]
                    if fs:
                        f=min(fs,key=lambda z:z['price']);return f"### 🛫 Cheapest flight\n\n**{f.get('airline')}** — {money(f.get('price'),f.get('currency','USD'))}."
                return render(ctx)

            req=normalize_request(route,base)
            # For destinations without their own commercial airport, resolve a practical airport code.
            if not req.get('destinationAirport') and req.get('destinationCity'):
                placeholder.info(f"⚡ Resolving flight airport for {req['destinationCity']}…")
                resolver=ChatGroq(model='openai/gpt-oss-20b',temperature=0,max_tokens=20)
                rr=await resolver.ainvoke([HumanMessage(content=f"Return ONLY one official uppercase 3-letter IATA code for the nearest practical commercial passenger airport serving {req['destinationCity']}. If the city has no airport, return the nearest practical airport. No explanation." )])
                m=re.search(r"\b[A-Z]{3}\b",(rr.content or '').upper())
                if m:req['destinationAirport']=m.group(0)

            if action in ('UPDATE','PLAN') and not ctx and action=='UPDATE': action='PLAN'
            ok,err=valid(req)
            if not ok:return f"## ⚠️ {err}"

            if action=='COMPARE':
                if not ctx:return "## 🧭 No active trip\n\nStart with a complete trip before comparing destinations."
                transport=await stack.enter_async_context(sse_client(server_url));session=await stack.enter_async_context(ClientSession(transport[0],transport[1]))
                placeholder.info(f"⚡ Fetching live {req.get('destinationCity')} data…")
                candidate=await get_trip(session,req);st.session_state.comparison_trips[req.get('destinationCity','Unknown')]=candidate;return compare_stats(ctx,candidate)

            transport=await stack.enter_async_context(sse_client(server_url));session=await stack.enter_async_context(ClientSession(transport[0],transport[1]))
            placeholder.info(f"⚡ Fetching live {req.get('destinationCity')} data…")
            trip=await get_trip(session,req)
            if trip.get('planningBlocked'):return f"## ⚠️ Trip cannot be planned\n\n{trip.get('error','Unknown error')}"
            st.session_state.active_trip=trip
            return render(trip)
        except Exception as exc:
            return f"## ❌ Something went wrong\n\n`{type(exc).__name__}: {exc}`"


for message in st.session_state.messages:
    with st.chat_message(message['role']): st.markdown(message['content'])

if prompt:=st.chat_input("Try: Chennai → Madurai, Aug 20–25, 1 traveler"):
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'):st.markdown(prompt)
    with st.chat_message('assistant'):
        response=asyncio.run(run_turn(prompt,st.empty()))
        st.markdown(response);st.session_state.messages.append({'role':'assistant','content':response})
