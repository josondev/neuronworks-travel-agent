import streamlit as st
import asyncio
import os
import json
import nest_asyncio
import ast
import urllib.request
from contextlib import AsyncExitStack
from pydantic import create_model, Field
from datetime import datetime
from typing import AsyncGenerator, Dict, Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool

nest_asyncio.apply()

# =============================================================================
# TIMEOUT CONSTANTS — tuned for Render free-tier cold starts
# =============================================================================
MCP_CONNECT_TIMEOUT = 60
MCP_TOOL_TIMEOUT = 45
LLM_INVOKE_TIMEOUT = 90

# =============================================================================
# UI & STYLING
# =============================================================================
st.set_page_config(page_title="Neuronworks Travel Agent", page_icon="✈️", layout="wide")

st.markdown("""
<style>
/* ===== DARK MODE (default) ===== */
:root {
    --bg: #080b14;
    --bg-secondary: rgba(15, 23, 42, 0.82);
    --border: rgba(255, 255, 255, 0.10);
    --text-primary: #f1f5f9;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --hero-bg: linear-gradient(135deg, rgba(37, 99, 235, 0.28), rgba(124, 58, 237, 0.22));
    --pill-bg: rgba(255, 255, 255, 0.09);
    --pill-text: #e2e8f0;
    --status-bg: rgba(30, 41, 59, 0.6);
    --info-bg: rgba(37, 99, 235, 0.15);
    --info-border: rgba(37, 99, 235, 0.4);
    --info-text: #93c5fd;
    --input-bg: rgba(15, 23, 42, 0.9);
    --input-border: rgba(255, 255, 255, 0.15);
    --input-focus-border: rgba(99, 102, 241, 0.6);
    --input-placeholder: #64748b;
}

/* ===== LIGHT MODE ===== */
@media (prefers-color-scheme: light) {
    :root {
        --bg: #f8fafc;
        --bg-secondary: rgba(255, 255, 255, 0.95);
        --border: rgba(0, 0, 0, 0.12);
        --text-primary: #0f172a;
        --text-secondary: #334155;
        --text-muted: #64748b;
        --hero-bg: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(124, 58, 237, 0.08));
        --pill-bg: rgba(0, 0, 0, 0.06);
        --pill-text: #334155;
        --status-bg: rgba(241, 245, 249, 0.95);
        --info-bg: rgba(37, 99, 235, 0.08);
        --info-border: rgba(37, 99, 235, 0.25);
        --info-text: #1d4ed8;
        --input-bg: rgba(255, 255, 255, 0.95);
        --input-border: rgba(0, 0, 0, 0.15);
        --input-focus-border: rgba(99, 102, 241, 0.6);
        --input-placeholder: #94a3b8;
    }
}

.stApp {
    background: radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.42), transparent 34%),
                radial-gradient(circle at 90% 10%, rgba(124, 58, 237, 0.36), transparent 32%),
                var(--bg);
}
@media (prefers-color-scheme: light) {
    .stApp {
        background: radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.08), transparent 34%),
                    radial-gradient(circle at 90% 10%, rgba(124, 58, 237, 0.06), transparent 32%),
                    var(--bg);
    }
}

.block-container { max-width: 1180px; padding-top: 5rem; padding-bottom: 6rem; }

.hero {
    padding: 26px 28px; border-radius: 22px; margin-bottom: 18px;
    background: var(--hero-bg); border: 1px solid var(--border);
}
.hero h1 { margin: 0; color: var(--text-primary); font-size: 2.2rem; }
.hero p { margin: 8px 0; color: var(--text-secondary); }
.pill {
    display: inline-block; padding: 5px 11px; border-radius: 999px;
    background: var(--pill-bg); color: var(--pill-text);
    font-size: 0.75rem; border: 1px solid var(--border);
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    border: 1px solid var(--border); border-radius: 18px;
    padding: 1rem 1.1rem; margin: 0.7rem 0; background: var(--bg-secondary);
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4,
div[data-testid="stChatMessage"] strong,
div[data-testid="stChatMessage"] em,
div[data-testid="stChatMessage"] code {
    color: var(--text-primary) !important;
}

/* Tool status widgets */
div[data-testid="stStatus"] {
    border: 1px solid var(--border); border-radius: 12px;
    margin: 0.5rem 0; background: var(--status-bg);
}
div[data-testid="stStatus"] summary {
    padding: 0.5rem 1rem; font-weight: 500; color: var(--text-muted);
}
div[data-testid="stStatus"] p,
div[data-testid="stStatus"] span,
div[data-testid="stStatus"] pre,
div[data-testid="stStatus"] code {
    color: var(--text-primary) !important;
}

/* Info/thinking alerts */
div[data-testid="stAlert"] {
    background: var(--info-bg) !important;
    border: 1px solid var(--info-border) !important;
    border-radius: 12px !important;
}
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span {
    color: var(--info-text) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary); border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}

/* ===== COMPOSER BAR (text area + send button) ===== */
div[data-testid="stTextArea"] > label {
    display: none !important;
}
textarea[data-testid="stTextArea"] {
    color: var(--text-primary) !important;
    background: var(--input-bg) !important;
    border: 1.5px solid var(--input-border) !important;
    border-radius: 16px !important;
    padding: 0.85rem 1rem !important;
    font-size: 1rem !important;
    line-height: 1.5 !important;
    resize: none !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease !important;
}
textarea[data-testid="stTextArea"]:hover {
    border-color: rgba(99, 102, 241, 0.35) !important;
}
textarea[data-testid="stTextArea"]:focus {
    border-color: var(--input-focus-border) !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.16), 0 2px 14px rgba(0, 0, 0, 0.14) !important;
    outline: none !important;
}
textarea[data-testid="stTextArea"]::placeholder {
    color: var(--input-placeholder) !important;
    font-style: italic !important;
}

/* Send button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    height: 68px !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.45) !important;
    opacity: 0.95 !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<span class="pill">● LIVE MCP · STREAMING · SEMANTIC ROUTING</span>
<h1>✈️ Neuronworks Travel Agent</h1>
<p>Flights · Hotels · Places · Restaurants · Weather · Budget · Currency</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR CONFIG
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    server_url = st.text_input("MCP Server URL", value="https://neuronworks-travel-agent.onrender.com/sse")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("⚠️ Set OPENROUTER_API_KEY env variable.")
        st.stop()

    st.success("✅ Connected to OpenRouter")
    st.caption("Model: Llama 3.3 70B Instruct (Streaming)")

    if st.button("🗑️ Clear Chat & Memory"):
        st.session_state.messages = []
        st.session_state.trip_data = {}
        st.rerun()

# =============================================================================
# HELPERS
# =============================================================================
_mcp_tool_schemas: dict = {}


def create_pydantic_model_from_schema(name, schema):
    fields = {}
    if "properties" in schema:
        required_fields = schema.get("required", [])
        for field_name, field_info in schema["properties"].items():
            field_type = str
            if field_info.get("type") == "number": field_type = float
            elif field_info.get("type") == "integer": field_type = int
            elif field_info.get("type") == "boolean": field_type = bool
            is_required = field_name in required_fields
            desc = field_info.get("description", "")
            if len(desc) > 100: desc = desc[:100] + "..."
            fields[field_name] = (field_type, Field(description=desc, default=... if is_required else None))
    return create_model(f"{name}Input", **fields)


def normalize_tool_args(tool_name: str, args: dict) -> dict:
    """Coerce LLM-generated arguments to match the MCP tool's expected JSON schema."""
    schema = _mcp_tool_schemas.get(tool_name, {})
    properties = schema.get("properties", {})
    normalized = dict(args)

    for key, value in list(normalized.items()):
        if key not in properties:
            continue
        expected_type = properties[key].get("type", "")

        if expected_type == "array":
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        normalized[key] = parsed; continue
                except (json.JSONDecodeError, TypeError): pass
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        normalized[key] = parsed; continue
                except (ValueError, SyntaxError): pass
                normalized[key] = [value]
        elif expected_type == "integer" and isinstance(value, str):
            try: normalized[key] = int(value)
            except ValueError: pass
        elif expected_type == "number" and isinstance(value, str):
            try: normalized[key] = float(value)
            except ValueError: pass
        elif expected_type == "boolean" and isinstance(value, str):
            normalized[key] = value.lower() in ("true", "1", "yes")

    return normalized


def clean_tool_output(tool_name: str, raw_text: str) -> str:
    """Aggressively filters garbage from place/attraction results."""
    tool_lower = tool_name.lower()
    if not any(kw in tool_lower for kw in ['place', 'attraction', 'search', 'poi', 'location']):
        return raw_text
    try:
        data = json.loads(raw_text)
        items, wrapper_key = [], None
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ['results', 'places', 'data', 'items', 'attractions', 'locations', 'candidates']:
                if key in data and isinstance(data[key], list):
                    items, wrapper_key = data[key], key; break
            if not items:
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        items, wrapper_key = v, k; break
        if not items: return raw_text

        good_types = {
            'tourist_attraction', 'museum', 'art_gallery', 'park', 'national_park',
            'zoo', 'aquarium', 'amusement_park', 'theme_park', 'water_park',
            'place_of_worship', 'church', 'mosque', 'hindu_temple', 'buddhist_temple',
            'synagogue', 'gurdwara', 'shrine', 'temple', 'historical_landmark',
            'monument', 'memorial', 'natural_feature', 'beach', 'lake', 'river',
            'waterfall', 'mountain', 'hill', 'viewpoint', 'observatory', 'planetarium',
            'science_center', 'restaurant', 'cafe', 'bar', 'food', 'shopping_mall',
            'market', 'department_store', 'movie_theater', 'night_club', 'casino',
            'lodging', 'hotel', 'resort_hotel', 'guest_house', 'hostel', 'stadium', 'arena',
        }
        bad_types = {
            'bus_station', 'bus_stop', 'train_station', 'subway_station', 'taxi_stand',
            'parking', 'car_rental', 'car_repair', 'gas_station', 'hospital', 'doctor',
            'dentist', 'pharmacy', 'clinic', 'school', 'university', 'police',
            'fire_station', 'post_office', 'courthouse', 'bank', 'atm', 'accounting',
            'real_estate_agency', 'electrician', 'plumber', 'hardware_store',
            'grocery_or_supermarket', 'convenience_store', 'gym', 'spa', 'beauty_salon',
            'laundry', 'funeral_home', 'cemetery', 'veterinary_care', 'storage',
            'local_government_office', 'city_hall', 'point_of_interest', 'establishment', "selai",
        }
        deny_words = [
            'water works', 'waterworks', 'water tank', 'water supply', 'sewage', 'drainage',
            'pump house', 'bus stop', 'bus stand', 'bus depot', 'railway station',
            'railway crossing', 'metro station', 'parking', 'car shelter', 'petrol pump',
            'gas station', 'toll booth', 'traffic signal', 'flyover', 'underpass', 'bypass',
            'junction', 'roundabout', 'police station', 'fire station', 'post office',
            'court', 'collector office', 'government office', 'nagar', 'colony', 'layout',
            'extension', 'township', 'housing board', 'apartment', 'slum', 'ward', 'block',
            'sector', 'bank', 'atm', 'factory', 'warehouse', 'godown', 'hospital', 'clinic',
            'pharmacy', 'school', 'college', 'statue', 'pillar', 'fountain', 'gate', 'tower',
            'bridge', 'dam', 'well', 'borewell', 'playground', 'maidan', 'road', 'street',
            'lane', 'highway', 'expressway', 'salai', 'theru', 'cemetery', 'graveyard',
            'prison', 'jail', 'military', 'cantonment', 'power station', 'substation',
            'telephone exchange', 'water board', "selai",
        ]

        cleaned = []
        for item in items:
            if isinstance(item, dict):
                name = str(item.get('name', '') or item.get('title', '') or '').lower()
                types = set()
                for key in ['types', 'categories', 'type', 'category', 'tags', 'primary_type']:
                    val = item.get(key)
                    if isinstance(val, list): types.update(str(t).lower() for t in val)
                    elif isinstance(val, str): types.add(val.lower())
                if types & good_types: cleaned.append(item); continue
                if types & bad_types: continue
                if any(d in name for d in deny_words): continue
                addr = str(item.get('formatted_address', '') or item.get('address', '') or '').lower()
                if any(d in addr for d in ['nagar', 'colony', 'layout', 'extension', 'township']): continue
                cleaned.append(item)
            else:
                cleaned.append(item)

        if wrapper_key:
            data[wrapper_key] = cleaned
            return json.dumps(data)
        return json.dumps(cleaned)
    except Exception:
        return raw_text


def beautify_output(text: str) -> str:
    if not text: return text
    text = text.replace("### Summary:", "### 📋 Summary:")
    text = text.replace("### Itinerary:", "### 🗓️ Itinerary:")
    text = text.replace("### Budget:", "### 💰 Budget:")
    text = text.replace("### Disclaimer:", "### ⚠️ Disclaimer:")
    return text


def synthesize_from_trip_data(trip_data: dict) -> str:
    """When the LLM returns empty after tool calls, build a response from cached data."""
    if not trip_data:
        return "I gathered data but couldn't generate a summary. Please try again."

    lines = ["## 📋 Trip Summary (auto-generated from tool data)\n"]

    for tool_name, data in trip_data.items():
        lines.append(f"### 🔧 {tool_name}")
        if isinstance(data, dict):
            if 'total_budget' in data:
                lines.append(f"- **Total Budget Estimate:** {data.get('currency', 'USD')} {data['total_budget']}")
                breakdown = data.get('breakdown', {})
                for k, v in breakdown.items():
                    lines.append(f"  - {k.replace('_', ' ').title()}: {data.get('currency', 'USD')} {v}")
            else:
                preview = json.dumps(data, separators=(',', ':'))[:500]
                lines.append(f"```\n{preview}\n```")
        elif isinstance(data, list):
            for item in data[:3]:
                if isinstance(item, dict):
                    name = item.get('name', item.get('airline', 'Unknown'))
                    price = item.get('price', '')
                    if price:
                        lines.append(f"- **{name}**: {item.get('currency', 'USD')} {price}")
                    else:
                        lines.append(f"- **{name}**")
        else:
            lines.append(f"```\n{str(data)[:500]}\n```")
        lines.append("")

    lines.append("\n⚠️ *This summary was auto-generated because the model couldn't produce a response. Prices and availability are subject to change.*")
    return "\n".join(lines)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================
current_date = datetime.now().strftime("%Y-%m-%d")

SYSTEM_PROMPT_TEMPLATE = """
You are an expert, factual AI Travel Agent. Your goal is to plan realistic, bookable trips using **only** real-time data from your tools.

### 📅 CURRENT CONTEXT
- **Today's Date:** {current_date}
- **Time Awareness:** When the user asks for "next Friday" or "in 2 days", calculate the exact date relative to {current_date}.

### 🛡️ CRITICAL RULES (DO NOT BREAK)
### 🛡️ THE "ZERO HALLUCINATION" PROTOCOL
1. **TRUTH OVER PLEASING:** If a tool returns no results (e.g., "No flights found"), you MUST tell the user: *"I could not find flights for these dates."* Do NOT invent a flight to make the user happy.
   **PRICE INTEGRITY:** - You must report the **EXACT PRICE** returned by the `search_flights` and Other tools.
   - **DO NOT LOWER THE PRICE** TO FIT THE USER'S "BUDGET" REQUEST. IF THE FLIGHT'S COST IS $100 AND THE USER WANTS "CHEAP", TELL THEM THE FLIGHT IS $100. DON'T INVENT A $10 FLIGHT.
2. **PRICING HONESTY:** - **NEVER** invent a specific price (e.g., "$119") if the tool didn't provide it. 
   - **EXCEPTION:** If the hotel tool returns a list of hotels but NO prices (or obvious mock prices), you may provide a **market estimate range** based on the hotel's tier (e.g., *"Typically $150-$200/night for a 5-star hotel in this city"*), but you MUST label it as an "Estimate".
3. **CURRENCY:** Keep the currency as returned by the tool (USD/EUR/INR). Do not convert unless explicitly asked.

### 🛠️ TOOL-SPECIFIC INSTRUCTIONS

#### 1. ✈️ FLIGHTS (`search_flights`)
- **CRITICAL:** The API fails if you send city names. You **MUST** convert them to 3-letter IATA codes.
  - "New York" -> `JFK` or `EWR`
  - "Paris" -> `CDG` or `ORY`
  - "London" -> `LHR` or `LGW`
  - "Madurai" -> `IXM`
  - "Chennai" -> `MAA`
  - *Internal Knowledge:* Use your training data to find codes for other cities.
- **DATES:** Format strictly as `YYYY-MM-DD`.

#### 2. 🏨 HOTELS (`Google Hotels`)
- **INPUT:** Send the full city name (e.g., "Paris").
- **ANALYSIS:** - If the tool returns hotels with names like "Taj", "Oberoi", "Hilton", treat them as **Luxury**.
  - If names contain "Inn", "Guest House", "Hostel", treat them as **Budget**.
  - **Budgeting:** If the API price seems fake (e.g., all hotels are exactly $80), use the hotel's category to estimate a realistic budget for the user.

#### 3. 🎡 PLACES (`search_places`)
- **STRICT CATEGORIES:** You may ONLY use these values for the `category` argument:
  - `tourist_attractions` (Museums, monuments)
  - `restaurants` (Food, dining)
  - `entertainment` (Nightlife, theaters)
  - `nature` (Parks, beaches)
  - `shopping` (Malls, markets)
  - `religion` (Temples, churches, mosques)
- **RADIUS:** Default to 5000 (5km) for city center, or 20000 (20km) if the user asks for "nearby" spots.

#### 4. 💰 BUDGET (`calculate_trip_budget`)
- **EXECUTION:** Call this tool **LAST**, only on the *first* full plan for a trip — not on every follow-up.
- **⚠️ CRITICAL ARGUMENT FORMAT:**
  - `destinations` MUST be a JSON **array** of strings, e.g. `["Goa"]` or `["Goa", "Pondicherry"]`. NEVER send a plain string like `"Goa"`. NEVER send a stringified list like `"['Goa']"`.
  - `duration` MUST be an **integer** (e.g. `4`), not a string `"4"`.
  - `travelers` MUST be an **integer** (e.g. `1`), not a string `"1"`.
- **VALID `budgetLevel` VALUES:** the tool only accepts exactly `budget`, `mid-range`, or `luxury` (nothing else — e.g. NOT `"low"`, `"cheap"`). Map the user's wording: "cheap/low/minimum" -> `budget`, "moderate/comfortable" -> `mid-range`, "luxury/high-end" -> `luxury`. If you send an invalid value, the tool silently falls back to `mid-range` pricing, which will be wrong.
- **IMPORTANT LIMITATION:** this tool does NOT accept real flight/hotel price as input — it only returns a generic estimate for the chosen `budgetLevel`. Do NOT claim you "fed it the real price." Instead, report the tool's estimate labeled as "Generic estimate," and **separately** compute and clearly label a "Actual total (from real data found)" by summing the exact flight price + (hotel nightly price × nights) + a stated daily-expense estimate. When the user asks you to minimize cost, the actual total is what should change — the generic tool estimate will not.
- **DO NOT RETRY ON ERROR:** If this tool returns an error, do NOT call it again with the same or similar arguments. Report the error to the user and move on.

### 🔁 FOLLOW-UP QUESTIONS (DO NOT RE-CALL TOOLS UNNECESSARILY)
- Prior tool results for this conversation are included below under `PRIOR TRIP DATA`, if any exist. Treat that as ground truth.
- If the user's follow-up can be answered by re-reasoning over `PRIOR TRIP DATA` (e.g. "pick the cheapest hotel from that list," "minimize cost," "which one is best for families") — DO NOT call any tool again. Just re-analyze the existing data and answer directly.
- Only call a tool again if the user asks for something the existing data cannot answer — a different city, different dates, a different category of place, or explicitly asks you to "search again" / "check for more/cheaper options."

### 📝 OUTPUT FORMAT
1. **Summary:** A quick breakdown of flight options and hotel recommendations.
2. **Itinerary:** A day-by-day plan using the specific *Attractions* found by `search_places`.
3. **Budget:** Both the generic tool estimate and the actual computed total (see above).
4. **Disclaimer:** "Prices and availability are subject to change."

{prior_data_section}
"""


# =============================================================================
# STREAMING AGENT (Async Generator)
# =============================================================================
async def run_agent_streaming(
    chat_history: list,
    trip_data: dict
) -> AsyncGenerator[Dict[str, Any], None]:

    async with AsyncExitStack() as stack:
        try:
            # --- Warm up Render free-tier server (cold start recovery) ---
            yield {"type": "thinking", "message": "☀️ Waking up MCP server (may take 30-60s on first use)..."}
            try:
                warmup_url = server_url.replace("/sse", "/health") if "/sse" in server_url else server_url
                req = urllib.request.Request(warmup_url, method="GET")
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, urllib.request.urlopen, req),
                    timeout=MCP_CONNECT_TIMEOUT
                )
            except Exception:
                pass  # Health endpoint may not exist — SSE connect will still work

            # --- Connect with timeout ---
            yield {"type": "thinking", "message": "🔌 Connecting to MCP server..."}
            try:
                transport = await asyncio.wait_for(
                    stack.enter_async_context(sse_client(server_url)),
                    timeout=MCP_CONNECT_TIMEOUT
                )
                session = await asyncio.wait_for(
                    stack.enter_async_context(ClientSession(transport[0], transport[1])),
                    timeout=MCP_CONNECT_TIMEOUT
                )
                if hasattr(session, "initialize"):
                    await asyncio.wait_for(session.initialize(), timeout=MCP_CONNECT_TIMEOUT)
            except asyncio.TimeoutError:
                yield {"type": "error", "message": f"❌ MCP server connection timed out after {MCP_CONNECT_TIMEOUT}s. The Render free-tier server may need more time to wake up — try again in a moment."}
                return

            # --- Discover Tools ---
            yield {"type": "thinking", "message": "🔍 Discovering tools..."}
            try:
                mcp_tools = await asyncio.wait_for(session.list_tools(), timeout=MCP_CONNECT_TIMEOUT)
            except asyncio.TimeoutError:
                yield {"type": "error", "message": "❌ Timed out discovering tools from MCP server."}
                return

            langchain_tools = []
            for tool in mcp_tools.tools:
                _mcp_tool_schemas[tool.name] = tool.input_schema or {}
                input_model = create_pydantic_model_from_schema(tool.name, tool.input_schema)
                tool_desc = (tool.description or "")[:150]
                lc_tool = StructuredTool.from_function(
                    func=None, coroutine=lambda *a, **kw: None,
                    name=tool.name, description=tool_desc, args_schema=input_model
                )
                langchain_tools.append(lc_tool)

            if not langchain_tools:
                yield {"type": "error", "message": "❌ No tools discovered from MCP server."}
                return

            # --- Build Messages ---
            llm = ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct",
                api_key=api_key, base_url="https://openrouter.ai/api/v1",
                default_headers={"X-Title": "AI Travel Agent"},
                temperature=0,
                request_timeout=LLM_INVOKE_TIMEOUT,
                max_retries=2,
            )
            llm_with_tools = llm.bind_tools(langchain_tools)

            prior_data_section = ""
            if trip_data:
                pj = json.dumps(trip_data, separators=(',', ':'))
                if len(pj) > 3000: pj = pj[:3000] + "...[truncated]"
                prior_data_section = f"### PRIOR TRIP DATA\n{pj}"

            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                current_date=current_date, prior_data_section=prior_data_section
            )
            messages = [SystemMessage(content=system_prompt)]
            history = [m for m in chat_history if m["role"] in ("user", "assistant")][-4:]
            for m in history:
                cls = HumanMessage if m["role"] == "user" else AIMessage
                messages.append(cls(content=m["content"]))

            # --- Agent Loop ---
            max_iterations = 5
            tool_call_counts: dict = {}
            MAX_CALLS_PER_TOOL = 2
            had_tool_calls = False

            for iteration in range(max_iterations):
                yield {"type": "thinking", "message": f"🤔 Reasoning... (step {iteration + 1})"}

                try:
                    ai_msg = await asyncio.wait_for(
                        llm_with_tools.ainvoke(messages),
                        timeout=LLM_INVOKE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    yield {"type": "error", "message": f"❌ LLM response timed out after {LLM_INVOKE_TIMEOUT}s."}
                    return
                except Exception as e:
                    yield {"type": "error", "message": f"❌ LLM error: {str(e)}"}
                    return

                messages.append(ai_msg)

                # --- Parse tool calls from content if needed (OpenRouter fix) ---
                tool_calls_to_execute = list(ai_msg.tool_calls or [])

                if not tool_calls_to_execute and ai_msg.content:
                    content_stripped = ai_msg.content.strip()
                    if (content_stripped.startswith("[") and '"type": "function"' in content_stripped) or \
                       (content_stripped.startswith("{") and '"type": "function"' in content_stripped):
                        try:
                            parsed = json.loads(content_stripped)
                            if isinstance(parsed, dict): parsed = [parsed]
                            for tc in parsed:
                                if tc.get("type") == "function":
                                    fn = tc.get("function", {})
                                    tool_calls_to_execute.append({
                                        "id": tc.get("id", f"call_{iteration}_{len(tool_calls_to_execute)}"),
                                        "name": fn.get("name", ""),
                                        "args": json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else fn.get("arguments", {}),
                                    })
                            messages[-1] = AIMessage(content="", tool_calls=tool_calls_to_execute)
                        except (json.JSONDecodeError, KeyError, TypeError):
                            pass

                if not tool_calls_to_execute:
                    break

                had_tool_calls = True
                tool_errors = []

                # --- Execute each tool call ---
                for tool_call in tool_calls_to_execute:
                    tool_name = tool_call['name']

                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                    if tool_call_counts[tool_name] > MAX_CALLS_PER_TOOL:
                        skip_msg = f"Skipped {tool_name}: already called {MAX_CALLS_PER_TOOL} times this turn."
                        yield {"type": "tool_end", "name": tool_name, "success": False, "preview": skip_msg}
                        messages.append(ToolMessage(
                            tool_call_id=tool_call['id'], content=skip_msg, name=tool_name
                        ))
                        continue

                    raw_args = tool_call['args']
                    normalized_args = normalize_tool_args(tool_name, raw_args)
                    yield {"type": "tool_start", "name": tool_name, "args": normalized_args}

                    try:
                        result = await asyncio.wait_for(
                            session.call_tool(tool_name, arguments=normalized_args),
                            timeout=MCP_TOOL_TIMEOUT
                        )
                        content_text = (
                            result.content[0].text
                            if result.content and hasattr(result.content[0], 'text')
                            else str(result)
                        )
                        content_text = clean_tool_output(tool_name, content_text)

                        # Check if the tool itself returned an error in its response
                        is_tool_error = False
                        try:
                            parsed_result = json.loads(content_text)
                            if isinstance(parsed_result, dict):
                                services = parsed_result.get("services", {})
                                for svc_name, svc_data in services.items():
                                    if isinstance(svc_data, dict) and "error" in svc_data:
                                        is_tool_error = True
                                        tool_errors.append(f"{svc_name}: {svc_data['error']}")
                                if parsed_result.get("error"):
                                    is_tool_error = True
                                    tool_errors.append(parsed_result["error"])
                                if parsed_result.get("planningBlocked"):
                                    is_tool_error = True
                                    tool_errors.append(parsed_result.get("error", "Planning blocked"))
                        except (json.JSONDecodeError, TypeError):
                            pass

                        try: trip_data[tool_name] = json.loads(content_text)
                        except: trip_data[tool_name] = content_text

                        messages.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=content_text, name=tool_name
                        ))

                        preview = content_text[:300] + "..." if len(content_text) > 300 else content_text
                        if is_tool_error:
                            yield {"type": "tool_end", "name": tool_name, "success": False, "preview": f"⚠️ Partial: {preview}"}
                        else:
                            yield {"type": "tool_end", "name": tool_name, "success": True, "preview": preview}

                    except asyncio.TimeoutError:
                        err = f"Tool {tool_name} timed out after {MCP_TOOL_TIMEOUT}s"
                        tool_errors.append(err)
                        messages.append(ToolMessage(
                            tool_call_id=tool_call['id'], content=err, name=tool_name
                        ))
                        yield {"type": "tool_end", "name": tool_name, "success": False, "preview": err}

                    except Exception as e:
                        err = f"Error: {str(e)}"
                        tool_errors.append(err)
                        messages.append(ToolMessage(
                            tool_call_id=tool_call['id'], content=err, name=tool_name
                        ))
                        yield {"type": "tool_end", "name": tool_name, "success": False, "preview": err}

                # If there were tool errors, tell the LLM to work with partial data
                if tool_errors:
                    error_summary = "; ".join(tool_errors)
                    messages.append(SystemMessage(
                        content=f"Some tools returned partial errors: {error_summary}. "
                        "Use whatever data WAS successfully returned. Do NOT say 'functions are insufficient'. "
                        "Present the available data and clearly note which parts failed."
                    ))

            # =============================================================
            # FINAL ANSWER: 3-tier fallback
            # =============================================================
            full_response = ""

            # Attempt 1: Stream tokens
            try:
                async for chunk in llm_with_tools.astream(messages):
                    if chunk.content:
                        stripped = chunk.content.strip()
                        if stripped.startswith("[") and '"type": "function"' in stripped:
                            continue
                        if stripped.startswith("{") and '"type": "function"' in stripped:
                            continue
                        full_response += chunk.content
                        yield {"type": "token", "content": chunk.content}
            except Exception:
                pass

            # Attempt 2: If streaming yielded nothing, try ainvoke
            if not full_response.strip():
                yield {"type": "thinking", "message": "📝 Generating summary..."}
                try:
                    fallback_msg = await asyncio.wait_for(
                        llm_with_tools.ainvoke(messages),
                        timeout=LLM_INVOKE_TIMEOUT
                    )
                    if fallback_msg.content and fallback_msg.content.strip():
                        full_response = fallback_msg.content
                        for char in full_response:
                            yield {"type": "token", "content": char}
                except Exception:
                    pass

            # Attempt 3: If LLM still returned nothing, synthesize from tool data
            if not full_response.strip():
                if had_tool_calls and trip_data:
                    full_response = synthesize_from_trip_data(trip_data)
                    for char in full_response:
                        yield {"type": "token", "content": char}
                else:
                    yield {"type": "error", "message": "⚠️ The model returned an empty response. Try rephrasing your query."}
                    return

            yield {"type": "done", "full_response": beautify_output(full_response)}

        except asyncio.TimeoutError:
            yield {"type": "error", "message": "❌ Operation timed out. Please try again."}
        except Exception as e:
            error_str = str(e)
            if "413" in error_str or "rate_limit_exceeded" in error_str:
                msg = "⚠️ Context limit exceeded. Clear chat memory to continue."
            elif "401" in error_str or "Unauthorized" in error_str:
                msg = "❌ Invalid API key. Check your OPENROUTER_API_KEY."
            elif "429" in error_str or "Too Many Requests" in error_str:
                msg = "⚠️ Rate limited. Wait a moment and try again."
            elif "Connection" in error_str or "connect" in error_str.lower():
                msg = "❌ Network error. Check your internet connection and MCP server URL."
            else:
                msg = f"❌ Error: {error_str}"
            yield {"type": "error", "message": msg}


# =============================================================================
# CHAT UI — using st.text_area + st.button instead of st.chat_input
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trip_data" not in st.session_state:
    st.session_state.trip_data = {}

# Render history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input area: plain text_area + send button
input_col, btn_col = st.columns([6, 1])

with input_col:
    user_input = st.text_area(
        "Where do you want to go?",
        placeholder="e.g., Plan a 3-day trip from Chennai to Goa...",
        height=68,
        key="chat_input_area",
        label_visibility="collapsed",
    )

with btn_col:
    st.write("")  # spacer to align button with textarea
    send_clicked = st.button("Send ➤", use_container_width=True)

# Process on send or Enter (Streamlit reruns on any widget interaction,
# so we check if there's text and the button was clicked)
if send_clicked and user_input and user_input.strip():
    prompt = user_input.strip()

    # Clear the text area for next input
    st.session_state["chat_input_area"] = ""

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        async def consume_stream():
            accumulated = ""
            active_status = None

            async for event in run_agent_streaming(
                st.session_state.messages,
                st.session_state.trip_data
            ):
                etype = event["type"]

                if etype == "thinking":
                    response_placeholder.info(event["message"])

                elif etype == "tool_start":
                    if active_status is not None:
                        active_status.update(state="complete")
                    active_status = st.status(
                        f"🛠️ Calling `{event['name']}`...", expanded=False
                    )
                    active_status.json(event["args"])

                elif etype == "tool_end":
                    if active_status is not None:
                        label = f"✅ {event['name']}" if event["success"] else f"❌ {event['name']}"
                        state = "complete" if event["success"] else "error"
                        active_status.text(event["preview"])
                        active_status.update(label=label, state=state)
                        active_status = None

                elif etype == "token":
                    accumulated += event["content"]
                    response_placeholder.markdown(
                        beautify_output(accumulated) + "▌"
                    )

                elif etype == "done":
                    if active_status is not None:
                        active_status.update(state="complete")
                    response_placeholder.markdown(event["full_response"])
                    return event["full_response"]

                elif etype == "error":
                    if active_status is not None:
                        active_status.update(state="error")
                    response_placeholder.error(event["message"])
                    return event["message"]

            return accumulated

        final_response = asyncio.run(consume_stream())
        if final_response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response
            })

    st.rerun()
