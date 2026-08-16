import streamlit as st
import asyncio
import os
import json
import nest_asyncio
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
# UI & STYLING
# =============================================================================
st.set_page_config(page_title="Neuronworks Travel Agent", page_icon="✈️", layout="wide")

st.markdown("""
<style>
:root{--bg:#080b14;--border:rgba(255,255,255,.10);--muted:#94a3b8}
.stApp{background:radial-gradient(circle at 10% 0%,rgba(37,99,235,.42),transparent 34%),radial-gradient(circle at 90% 10%,rgba(124,58,237,.36),transparent 32%),var(--bg)}
.block-container{max-width:1180px;padding-top:5rem;padding-bottom:6rem}
.hero{padding:26px 28px;border-radius:22px;margin-bottom:18px;background:linear-gradient(135deg,rgba(37,99,235,.28),rgba(124,58,237,.22));border:1px solid var(--border)}
.hero h1{margin:0;color:#fff;font-size:2.2rem}.hero p{margin:8px 0;color:#cbd5e1}
.pill{display:inline-block;padding:5px 11px;border-radius:999px;background:rgba(255,255,255,.09);color:#e2e8f0;font-size:.75rem;border:1px solid var(--border)}
div[data-testid="stChatMessage"]{border:1px solid var(--border);border-radius:18px;padding:1rem 1.1rem;margin:.7rem 0;background:rgba(15,23,42,.82)}
div[data-testid="stStatus"]{border:1px solid var(--border);border-radius:12px;margin:0.5rem 0;background:rgba(30,41,59,0.6)}
div[data-testid="stStatus"] summary{padding:0.5rem 1rem;font-weight:500;color:#94a3b8}
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

# Store raw MCP schemas globally so we can normalize args against them
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
    """
    Coerce LLM-generated arguments to match the MCP tool's expected JSON schema.
    Fixes common mistakes:
      - string "Goa" → ["Goa"] when schema expects array
      - string "4" → 4 when schema expects integer/number
      - string "true" → True when schema expects boolean
    """
    schema = _mcp_tool_schemas.get(tool_name, {})
    properties = schema.get("properties", {})
    normalized = dict(args)

    for key, value in list(normalized.items()):
        if key not in properties:
            continue
        expected_type = properties[key].get("type", "")

        # String → Array coercion
        if expected_type == "array" and isinstance(value, str):
            normalized[key] = [value]
        elif expected_type == "array" and isinstance(value, list):
            pass  # already correct

        # String → Integer coercion
        elif expected_type == "integer" and isinstance(value, str):
            try:
                normalized[key] = int(value)
            except ValueError:
                pass

        # String → Number coercion
        elif expected_type == "number" and isinstance(value, str):
            try:
                normalized[key] = float(value)
            except ValueError:
                pass

        # String → Boolean coercion
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
                    items, wrapper_key = data[key], key
                    break
            if not items:
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        items, wrapper_key = v, k
                        break
        if not items:
            return raw_text

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
            'local_government_office', 'city_hall', 'point_of_interest', 'establishment',
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
            'telephone exchange', 'water board',
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
                if types & good_types:
                    cleaned.append(item); continue
                if types & bad_types:
                    continue
                if any(d in name for d in deny_words):
                    continue
                addr = str(item.get('formatted_address', '') or item.get('address', '') or '').lower()
                if any(d in addr for d in ['nagar', 'colony', 'layout', 'extension', 'township']):
                    continue
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
  - `destinations` MUST be a JSON **array** of strings, e.g. `["Goa"]` or `["Goa", "Pondicherry"]`. NEVER send a plain string like `"Goa"`.
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
            # --- Connect ---
            yield {"type": "thinking", "message": "🔌 Connecting to MCP server..."}
            transport = await stack.enter_async_context(sse_client(server_url))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            if hasattr(session, "initialize"):
                await session.initialize()

            # --- Discover Tools ---
            yield {"type": "thinking", "message": "🔍 Discovering tools..."}
            mcp_tools = await session.list_tools()
            langchain_tools = []

            for tool in mcp_tools.tools:
                # Store raw schema for argument normalization
                _mcp_tool_schemas[tool.name] = tool.input_schema or {}

                input_model = create_pydantic_model_from_schema(tool.name, tool.input_schema)
                tool_desc = (tool.description or "")[:150]
                lc_tool = StructuredTool.from_function(
                    func=None, coroutine=lambda *a, **kw: None,
                    name=tool.name, description=tool_desc, args_schema=input_model
                )
                langchain_tools.append(lc_tool)

            # --- Build Messages ---
            llm = ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct",
                api_key=api_key, base_url="https://openrouter.ai/api/v1",
                default_headers={"X-Title": "AI Travel Agent"}, temperature=0
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
            # Track how many times each tool has been called to prevent infinite retries
            tool_call_counts: dict = {}
            MAX_CALLS_PER_TOOL = 2

            for iteration in range(max_iterations):
                yield {"type": "thinking", "message": f"🤔 Reasoning... (step {iteration + 1})"}

                ai_msg = await llm_with_tools.ainvoke(messages)
                messages.append(ai_msg)

                # ---------------------------------------------------------
                # FIX: OpenRouter sometimes returns tool calls as JSON
                # string inside content instead of tool_calls array
                # ---------------------------------------------------------
                tool_calls_to_execute = list(ai_msg.tool_calls or [])

                if not tool_calls_to_execute and ai_msg.content:
                    content_stripped = ai_msg.content.strip()
                    if (content_stripped.startswith("[") and '"type": "function"' in content_stripped) or \
                       (content_stripped.startswith("{") and '"type": "function"' in content_stripped):
                        try:
                            parsed = json.loads(content_stripped)
                            if isinstance(parsed, dict):
                                parsed = [parsed]
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

                # Execute each tool call
                for tool_call in tool_calls_to_execute:
                    tool_name = tool_call['name']

                    # --- Per-tool retry cap ---
                    tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                    if tool_call_counts[tool_name] > MAX_CALLS_PER_TOOL:
                        skip_msg = f"Skipped {tool_name}: already called {MAX_CALLS_PER_TOOL} times this turn. Moving on."
                        yield {"type": "tool_end", "name": tool_name, "success": False, "preview": skip_msg}
                        messages.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=skip_msg, name=tool_name
                        ))
                        continue

                    # --- Normalize arguments against MCP schema ---
                    raw_args = tool_call['args']
                    normalized_args = normalize_tool_args(tool_name, raw_args)

                    yield {"type": "tool_start", "name": tool_name, "args": normalized_args}

                    try:
                        result = await session.call_tool(tool_name, arguments=normalized_args)
                        content_text = (
                            result.content[0].text
                            if result.content and hasattr(result.content[0], 'text')
                            else str(result)
                        )
                        content_text = clean_tool_output(tool_name, content_text)

                        try: trip_data[tool_name] = json.loads(content_text)
                        except: trip_data[tool_name] = content_text

                        messages.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=content_text, name=tool_name
                        ))
                        preview = content_text[:300] + "..." if len(content_text) > 300 else content_text
                        yield {"type": "tool_end", "name": tool_name, "success": True, "preview": preview}

                    except Exception as e:
                        err = f"Error: {str(e)}"
                        messages.append(ToolMessage(
                            tool_call_id=tool_call['id'], content=err, name=tool_name
                        ))
                        yield {"type": "tool_end", "name": tool_name, "success": False, "preview": err}

            # --- Stream Final Answer Token-by-Token ---
            full_response = ""
            async for chunk in llm_with_tools.astream(messages):
                if chunk.content:
                    stripped = chunk.content.strip()
                    if stripped.startswith("[") and '"type": "function"' in stripped:
                        continue
                    if stripped.startswith("{") and '"type": "function"' in stripped:
                        continue
                    full_response += chunk.content
                    yield {"type": "token", "content": chunk.content}

            yield {"type": "done", "full_response": beautify_output(full_response)}

        except Exception as e:
            error_str = str(e)
            if "413" in error_str or "rate_limit_exceeded" in error_str:
                msg = "⚠️ Context limit exceeded. Clear chat memory to continue."
            else:
                msg = f"❌ Error: {error_str}"
            yield {"type": "error", "message": msg}


# =============================================================================
# CHAT UI WITH STREAMING RENDERER
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trip_data" not in st.session_state:
    st.session_state.trip_data = {}

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Where do you want to go?"):
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
