import streamlit as st
import asyncio
import os
import json
import nest_asyncio
from contextlib import AsyncExitStack
from pydantic import create_model, Field
from datetime import datetime

from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool

# 1. Apply Async Patch for Streamlit
nest_asyncio.apply()

# 2. Page Config & UI (Your original beautiful styling)
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

/* Tool call status styling */
div[data-testid="stStatus"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    margin: 0.5rem 0;
    background: rgba(30, 41, 59, 0.6);
}
div[data-testid="stStatus"] summary {
    padding: 0.5rem 1rem;
    font-weight: 500;
    color: #94a3b8;
}
div[data-testid="stStatus"] div[data-testid="stMarkdown"] {
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<span class="pill">● LIVE MCP · SEMANTIC ROUTING</span>
<h1>✈️ Neuronworks Travel Agent</h1>
<p>Flights · Hotels · Places · Restaurants · Weather · Budget · Currency</p>
</div>
""", unsafe_allow_html=True)

# 3. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    server_url = st.text_input("MCP Server URL", value="https://neuronworks-travel-agent.onrender.com/sse")
    
    # Read API Key from environment variables
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        st.error("⚠️ Please set the OPENROUTER_API_KEY environment variable.")
        st.stop()
        
    st.success("✅ Connected to OpenRouter")
    st.caption("Model: Llama 3.3 70B Instruct")
    
    if st.button("🗑️ Clear Chat & Memory"):
        st.session_state.messages = []
        st.session_state.trip_data = {}
        st.rerun()

# --- HELPER: Convert JSON Schema to Pydantic (Token Optimized) ---
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
            
            fields[field_name] = (
                field_type, 
                Field(description=desc, default=... if is_required else None)
            )
    return create_model(f"{name}Input", **fields)

# --- 🛡️ AGGRESSIVE GARBAGE FILTER ---
def clean_tool_output(tool_name, raw_text):
    """Aggressively filters out garbage from place/attraction tool results."""
    tool_lower = tool_name.lower()
    if not any(keyword in tool_lower for keyword in ['place', 'attraction', 'search', 'poi', 'location']):
        return raw_text
    
    try:
        data = json.loads(raw_text)
        items, wrapper_key = [], None
        
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ['results', 'places', 'data', 'items', 'attractions', 'locations', 'candidates']:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    wrapper_key = key
                    break
            if not items:
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        items = v
                        wrapper_key = k
                        break
        
        if not items:
            return raw_text
        
        # Google Places types that ARE tourist attractions
        good_types = {
            'tourist_attraction', 'museum', 'art_gallery', 'park', 'national_park',
            'zoo', 'aquarium', 'amusement_park', 'theme_park', 'water_park',
            'place_of_worship', 'church', 'mosque', 'hindu_temple', 'buddhist_temple',
            'synagogue', 'gurdwara', 'shrine', 'temple',
            'historical_landmark', 'monument', 'memorial',
            'natural_feature', 'beach', 'lake', 'river', 'waterfall', 'mountain', 'hill',
            'viewpoint', 'observatory', 'planetarium', 'science_center',
            'restaurant', 'cafe', 'bar', 'food', 'meal_takeaway', 'meal_delivery',
            'shopping_mall', 'market', 'department_store', 'supermarket',
            'movie_theater', 'night_club', 'casino', 'amusement_center',
            'lodging', 'hotel', 'resort_hotel', 'guest_house', 'hostel',
            'stadium', 'arena', 'park', 'national_park', 'state_park', 'city_park',
            'campground', 'rv_park', 'tourist_attraction',
        }
        
        # Google Places types that are NOT tourist attractions
        bad_types = {
            'bus_station', 'bus_stop', 'train_station', 'subway_station', 'tram_station',
            'taxi_stand', 'parking', 'car_rental', 'car_repair', 'car_wash', 'car_dealer',
            'gas_station', 'petrol_station', 'fuel_station', 'electric_vehicle_charging_station',
            'hospital', 'doctor', 'dentist', 'pharmacy', 'clinic', 'health', 'physiotherapist',
            'school', 'university', 'college', 'library', 'day_care', 'kindergarten',
            'police', 'fire_station', 'post_office', 'courthouse', 'embassy', 'city_hall',
            'bank', 'atm', 'accounting', 'finance', 'insurance_agency', 'lawyer',
            'real_estate_agency', 'travel_agency', 'moving_company',
            'electrician', 'plumber', 'roofing_contractor', 'general_contractor', 'locksmith',
            'hardware_store', 'home_goods_store', 'furniture_store', 'electronics_store',
            'clothing_store', 'shoe_store', 'jewelry_store', 'florist', 'book_store',
            'grocery_or_supermarket', 'convenience_store', 'bakery', 'butcher', 'liquor_store',
            'gym', 'spa', 'beauty_salon', 'hair_care', 'laundry', 'dry_cleaning',
            'movie_rental', 'bicycle_store', 'pet_store', 'veterinary_care',
            'funeral_home', 'cemetery', 'crematorium',
            'local_government_office', 'storage', 'storage_rental', 'warehouse',
        }
        
        # Name-based deny list (comprehensive)
        deny_words = [
            # Infrastructure
            'water works', 'waterworks', 'water tank', 'water supply', 'water treatment',
            'sewage', 'drainage', 'pump house', 'pumping station', 'water board', 'water authority',
            # Transportation
            'bus stop', 'bus stand', 'bus station', 'bus depot', 'bus terminal', 'bus bay',
            'railway station', 'railway crossing', 'railway gate', 'railway track', 'railway colony',
            'metro station', 'metro rail', 'subway', 'tram stop', 'tramway',
            'airport', 'aerodrome', 'heliport', 'helipad', 'air strip',
            'parking', 'car park', 'car shelter', 'garage', 'car wash',
            'petrol pump', 'petrol bunk', 'gas station', 'fuel station', 'diesel', 'lpg',
            'toll', 'toll booth', 'toll plaza', 'toll gate', 'toll road',
            'signal', 'traffic signal', 'traffic light', 'traffic circle', 'traffic island',
            'flyover', 'underpass', 'bypass', 'overpass', 'interchange', 'junction',
            'roundabout', 'circle', 'chowk', 'square', 'crossing', 'intersection',
            # Government/Administrative
            'police station', 'police post', 'police booth', 'police line', 'police quarters',
            'fire station', 'fire brigade', 'fire office', 'fire control',
            'post office', 'head post office', 'sub post office', 'postal',
            'court', 'courthouse', 'district court', 'high court', 'sessions court', 'tribunal',
            'collector office', 'district office', 'taluk office', 'tehsil', 'revenue office',
            'government office', 'govt office', 'municipal office', 'corporation office',
            'passport office', 'immigration office', 'visa office', 'election office',
            'ration office', 'fair price', 'pds', 'public distribution',
            'employment exchange', 'job center', 'recruitment office',
            # Residential
            'nagar', 'colony', 'layout', 'extension', 'township', 'housing board',
            'housing society', 'apartment', 'flat', 'building', 'complex', 'enclave',
            'slum', 'chawl', 'basti', 'jhuggi', 'jhopri', 'jhuggi jhopri',
            'ward', 'block', 'sector', 'zone', 'division', 'phase', 'pocket',
            # Commercial (non-tourist)
            'bank', 'atm', 'cash', 'money', 'currency exchange', 'forex',
            'office', 'workspace', 'co-working', 'business center', 'business park',
            'factory', 'mill', 'plant', 'industry', 'industrial', 'industrial area',
            'warehouse', 'godown', 'depot', 'storage', 'cold storage',
            'shop', 'store', 'market', 'bazaar', 'mandi', 'wholesale',
            # Medical
            'hospital', 'clinic', 'dispensary', 'health center', 'health centre',
            'medical', 'pharmacy', 'chemist', 'drug store', 'medicine', 'medical store',
            'nursing home', 'maternity', 'diagnostic', 'lab', 'pathology', 'radiology',
            'blood bank', 'ambulance', 'mortuary',
            # Education (non-tourist)
            'school', 'college', 'university', 'institute', 'academy', 'coaching',
            'kindergarten', 'nursery', 'play school', 'day care', 'creche',
            'library', 'reading room', 'study center', 'study centre', 'tuition',
            # Random structures
            'statue', 'monument', 'memorial', 'pillar', 'column', 'obelisk',
            'fountain', 'sculpture', 'mural', 'wall painting', 'graffiti',
            'gate', 'gateway', 'arch', 'portal', 'entrance', 'exit',
            'tower', 'minaret', 'clock tower', 'water tower', 'mobile tower',
            'bridge', 'dam', 'barrage', 'weir', 'canal', 'lock', 'sluice',
            'well', 'borewell', 'hand pump', 'tube well', 'step well',
            'ground', 'playground', 'maidan', 'field', 'stadium', 'arena', 'sports complex',
            'gym', 'fitness', 'yoga', 'sports', 'swimming pool', 'pool',
            # Roads and paths
            'road', 'street', 'lane', 'alley', 'path', 'walkway', 'sidewalk', 'footpath',
            'highway', 'expressway', 'freeway', 'motorway', 'state highway', 'national highway',
            'salai', 'theru', 'marg', 'road', 'street', 'lane', 'path',
            # Other
            'cemetery', 'graveyard', 'burial ground', 'crematorium', 'smashan', 'shamshan',
            'slaughterhouse', 'abattoir', 'butcher', 'meat shop',
            'prison', 'jail', 'detention center', 'correctional facility',
            'military', 'army', 'navy', 'air force', 'cantonment', 'camp', 'barracks',
            'quarantine', 'isolation center', 'vaccination center', 'testing center',
            'electricity', 'power station', 'substation', 'transformer', 'grid station',
            'telephone exchange', 'telecom', 'bsnl', 'airtel', 'jio', 'vodafone',
            'sewage treatment', 'waste management', 'garbage dump', 'landfill',
            'water works', 'water board', 'water authority', 'water supply',
        ]
        
        cleaned_items = []
        for item in items:
            if isinstance(item, dict):
                # Get name from various possible fields
                name = str(item.get('name', '') or item.get('title', '') or item.get('display_name', '') or '').lower()
                
                # Get types/categories
                types = set()
                for key in ['types', 'categories', 'type', 'category', 'tags', 'place_types', 'primary_type']:
                    if key in item:
                        val = item[key]
                        if isinstance(val, list):
                            types.update([str(t).lower() for t in val])
                        elif isinstance(val, str):
                            types.add(val.lower())
                
                # Check if it's a good type (tourist attraction)
                if types & good_types:
                    cleaned_items.append(item)
                    continue
                
                # Check if it's a bad type (non-tourist)
                if types & bad_types:
                    continue
                
                # If no types available or generic types, check name against deny list
                if any(deny in name for deny in deny_words):
                    continue
                
                # Check address for residential keywords
                address = str(item.get('formatted_address', '') or item.get('address', '') or item.get('vicinity', '') or '').lower()
                if any(deny in address for deny in ['nagar', 'colony', 'layout', 'extension', 'township', 'housing']):
                    continue
                
                # If we get here, keep the item (conservative approach)
                cleaned_items.append(item)
            else:
                cleaned_items.append(item)
                
        if wrapper_key:
            data[wrapper_key] = cleaned_items
            return json.dumps(data)
        else:
            return json.dumps(cleaned_items)
            
    except Exception:
        return raw_text

# --- 🎨 BEAUTIFY OUTPUT ---
def beautify_output(text):
    """Enhances the LLM output for better display."""
    if not text:
        return text
    
    # Ensure proper spacing between sections
    text = text.replace("###", "\n###")
    
    # Add emojis to section headers if not already present
    text = text.replace("### Summary:", "### 📋 Summary:")
    text = text.replace("### Itinerary:", "### 🗓️ Itinerary:")
    text = text.replace("### Budget:", "### 💰 Budget:")
    text = text.replace("### Disclaimer:", "### ⚠️ Disclaimer:")
    
    return text

# --- SYSTEM PROMPT ---
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
- **VALID `budgetLevel` VALUES:** the tool only accepts exactly `budget`, `mid-range`, or `luxury` (nothing else — e.g. NOT `"low"`, `"cheap"`). Map the user's wording: "cheap/low/minimum" -> `budget`, "moderate/comfortable" -> `mid-range`, "luxury/high-end" -> `luxury`. If you send an invalid value, the tool silently falls back to `mid-range` pricing, which will be wrong.
- **IMPORTANT LIMITATION:** this tool does NOT accept real flight/hotel price as input — it only returns a generic estimate for the chosen `budgetLevel`. Do NOT claim you "fed it the real price." Instead, report the tool's estimate labeled as "Generic estimate," and **separately** compute and clearly label a "Actual total (from real data found)" by summing the exact flight price + (hotel nightly price × nights) + a stated daily-expense estimate. When the user asks you to minimize cost, the actual total is what should change — the generic tool estimate will not.

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

# --- CORE LOGIC ---
async def run_agent(chat_history, trip_data, chat_container):
    status_text = chat_container.empty()
    status_text.info("🔌 Connecting to Server...")
    
    async with AsyncExitStack() as stack:
        try:
            # 1. Connect
            transport = await stack.enter_async_context(sse_client(server_url))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            
            if hasattr(session, "initialize"):
                await session.initialize()
                
            status_text.info("✅ Connected! Discovering tools...")

            # 2. List Tools
            mcp_tools = await session.list_tools()
            langchain_tools = []

            for tool in mcp_tools.tools:
                input_model = create_pydantic_model_from_schema(tool.name, tool.input_schema)
                tool_desc = tool.description or ""
                if len(tool_desc) > 150: tool_desc = tool_desc[:150] + "..."
                
                lc_tool = StructuredTool.from_function(
                    func=None,
                    coroutine=lambda *args, **kwargs: None,
                    name=tool.name,
                    description=tool_desc,
                    args_schema=input_model
                )
                langchain_tools.append(lc_tool)
            
            status_text.info(f"🛠️ Found {len(langchain_tools)} tools. Thinking...")

            # 3. Initialize LLM (OpenRouter)
            llm = ChatOpenAI(
                model="meta-llama/llama-3.3-70b-instruct",
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={"X-Title": "AI Travel Agent"},
                temperature=0
            )
            llm_with_tools = llm.bind_tools(langchain_tools)

            # 4. Construct Messages
            prior_data_section = ""
            if trip_data:
                prior_json = json.dumps(trip_data, separators=(',', ':'))
                if len(prior_json) > 3000:
                    prior_json = prior_json[:3000] + "...[truncated]"
                prior_data_section = f"### PRIOR TRIP DATA\n{prior_json}"
            
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                current_date=current_date,
                prior_data_section=prior_data_section
            )

            messages = [SystemMessage(content=system_prompt)]
            history = [m for m in chat_history if m["role"] in ("user", "assistant")][-4:]
            for m in history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                else:
                    messages.append(AIMessage(content=m["content"]))

            # 5. Agent Loop
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                status_text.info(f"🤔 Thinking... (Step {iteration})")
                ai_msg = await llm_with_tools.ainvoke(messages)
                messages.append(ai_msg)

                if not ai_msg.tool_calls:
                    break 
                
                status_text.info(f"🛠️ Executing {len(ai_msg.tool_calls)} tool(s)...")
                
                for tool_call in ai_msg.tool_calls:
                    # Show tool call in a collapsible status widget
                    with st.status(f"🛠️ Calling: `{tool_call['name']}`", expanded=False) as status:
                        status.write(f"**Input:**")
                        status.json(tool_call['args'])
                        
                        try:
                            tool_result = await session.call_tool(tool_call['name'], arguments=tool_call['args'])
                            
                            if tool_result.content and hasattr(tool_result.content[0], 'text'):
                                content_text = tool_result.content[0].text
                            else:
                                content_text = str(tool_result)
                                
                            # 🚨 FILTER GARBAGE ATTRACTIONS BEFORE SAVING OR SENDING TO LLM
                            content_text = clean_tool_output(tool_call['name'], content_text)
                                
                            try:
                                trip_data[tool_call['name']] = json.loads(content_text)
                            except:
                                trip_data[tool_call['name']] = content_text
                            
                            # Show result in status
                            status.write(f"**Output:**")
                            display_text = content_text[:1000] + "..." if len(content_text) > 1000 else content_text
                            status.text(display_text)
                            
                            status.update(label=f"✅ {tool_call['name']} completed", state="complete")
                            
                            messages.append(ToolMessage(
                                tool_call_id=tool_call['id'],
                                content=content_text,
                                name=tool_call['name']
                            ))
                        except Exception as e:
                            error_msg = f"Error executing tool {tool_call['name']}: {str(e)}"
                            status.update(label=f"❌ {tool_call['name']} failed", state="error")
                            messages.append(ToolMessage(
                                tool_call_id=tool_call['id'],
                                content=error_msg,
                                name=tool_call['name']
                            ))
            
            status_text.empty()
            return beautify_output(ai_msg.content)

        except Exception as e:
            status_text.empty()
            error_str = str(e)
            if "413" in error_str or "Request too large" in error_str or "rate_limit_exceeded" in error_str:
                return "⚠️ **Context Limit Exceeded:** The conversation history and tool data became too large. Please click **'🗑️ Clear Chat & Memory'** in the sidebar to start a fresh trip."
            return f"An error occurred: {error_str}"

# --- UI: Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "trip_data" not in st.session_state:
    st.session_state.trip_data = {}

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Where do you want to go? (e.g., Plan a trip to Paris next Friday)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = asyncio.run(run_agent(st.session_state.messages, st.session_state.trip_data, st.empty()))
        
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
