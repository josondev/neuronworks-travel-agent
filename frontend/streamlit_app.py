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
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool

st.markdown("""
<style>
:root{--bg:#080b14;--border:rgba(255,255,255,.10);--muted:#94a3b8}
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
<span class="pill">● LIVE MCP · SEMANTIC ROUTING</span>
<h1>✈️ Neuronworks Travel Agent</h1>
<p>Flights · Hotels · Places · Restaurants · Weather · Budget · Currency</p>
</div>
""", unsafe_allow_html=True)


# 1. Apply Async Patch for Streamlit
nest_asyncio.apply()

# 3. Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    server_url = st.text_input("MCP Server URL", value="https://neuronworks-travel-agent.onrender.com/sse")
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = st.text_input("Groq API Key", type="password")
    
    if not api_key:
        st.warning("⚠️ Please enter a Groq API Key to continue.")
        st.stop()
    
    os.environ["GROQ_API_KEY"] = api_key
    st.success("✅ Ready to fly!")

# --- HELPER: Convert JSON Schema to Pydantic ---
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
            fields[field_name] = (
                field_type, 
                Field(description=field_info.get("description", ""), default=... if is_required else None)
            )
    return create_model(f"{name}Input", **fields)

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
            
            # Initialize session (required for modern MCP clients)
            if hasattr(session, "initialize"):
                await session.initialize()
                
            status_text.info("✅ Connected! Discovering tools...")

            # 2. List Tools
            mcp_tools = await session.list_tools()
            langchain_tools = []

            for tool in mcp_tools.tools:
                input_model = create_pydantic_model_from_schema(tool.name, tool.input_schema)
                lc_tool = StructuredTool.from_function(
                    func=None,
                    coroutine=lambda *args, **kwargs: None, # Dummy, we call session directly
                    name=tool.name,
                    description=tool.description,
                    args_schema=input_model
                )
                langchain_tools.append(lc_tool)
            
            status_text.info(f"🛠️ Found {len(langchain_tools)} tools. Thinking...")

            # 3. Initialize LLM
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
            llm_with_tools = llm.bind_tools(langchain_tools)

            # 4. Construct Messages
            prior_data_section = ""
            if trip_data:
                prior_json = json.dumps(trip_data, indent=2)
                if len(prior_json) > 4000:
                    prior_json = prior_json[:4000] + "\n... [truncated] ..."
                prior_data_section = f"### PRIOR TRIP DATA\n```json\n{prior_json}\n```"
            
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                current_date=current_date,
                prior_data_section=prior_data_section
            )

            messages = [SystemMessage(content=system_prompt)]
            
            # Add chat history (limit to last 10 turns to save tokens)
            history = [m for m in chat_history if m["role"] in ("user", "assistant")][-10:]
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

                # If no tool calls, we have our final answer
                if not ai_msg.tool_calls:
                    break 
                
                status_text.info(f"🛠️ Executing {len(ai_msg.tool_calls)} tool(s)...")
                
                for tool_call in ai_msg.tool_calls:
                    status_text.info(f"🛠️ Executing: `{tool_call['name']}`")
                    try:
                        # Execute Tool directly via MCP session
                        tool_result = await session.call_tool(tool_call['name'], arguments=tool_call['args'])
                        
                        if tool_result.content and hasattr(tool_result.content[0], 'text'):
                            content_text = tool_result.content[0].text
                        else:
                            content_text = str(tool_result)
                            
                        # Save to trip_data cache for follow-ups
                        try:
                            trip_data[tool_call['name']] = json.loads(content_text)
                        except:
                            trip_data[tool_call['name']] = content_text
                            
                        messages.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=content_text,
                            name=tool_call['name']
                        ))
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_call['name']}: {str(e)}"
                        messages.append(ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=error_msg,
                            name=tool_call['name']
                        ))
            
            status_text.empty()
            return ai_msg.content

        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            return f"An error occurred: {str(e)}"

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
