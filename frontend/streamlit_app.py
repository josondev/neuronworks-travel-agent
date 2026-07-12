import streamlit as st
import asyncio
import os
import json
import nest_asyncio
from contextlib import AsyncExitStack
from pydantic import create_model, Field
from datetime import datetime 

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool

st.markdown("""
<style>
  /* App background */
  .stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  }

  section.main > div {
    max-width: 100%;
    padding: 1rem;
  }

  /* Base chat message box styling (common) */
  div[data-testid="stChatMessage"]{
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.12);
  }

  /* DARK MODE */
  @media (prefers-color-scheme: dark) {
    div[data-testid="stChatMessage"]{
      background: #0b0b0b !important;
    }

    /* Target the actual rendered text inside chat messages */
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] div{
      color: #f7fafc !important;
    }
  }

  /* LIGHT MODE */
  @media (prefers-color-scheme: light) {
    div[data-testid="stChatMessage"]{
      background: #ffffff !important;
    }

    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] div{
      color: #111827 !important;
    }
  }
</style>
""", unsafe_allow_html=True)

# 1. Apply Async Patch
nest_asyncio.apply()

# 2. Page Config
st.set_page_config(page_title="AI Travel Agent", page_icon="✈️")
st.title("✈️ AI Travel Agent")
st.caption("Powered by Neuronworks.ai & MCP")

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
        for field_name, field_info in schema["properties"].items():
            field_type = str 
            if field_info.get("type") == "number": field_type = float
            elif field_info.get("type") == "integer": field_type = int
            elif field_info.get("type") == "boolean": field_type = bool
            
            fields[field_name] = (field_type, Field(description=field_info.get("description", "")))
    return create_model(f"{name}Input", **fields)

# --- SYSTEM PROMPT (The Anti-Hallucination Guard) ---
# --- SYSTEM PROMPT (Strictly aligned with your Service Code) ---
# --- GET CURRENT DATE ---
current_date = datetime.now().strftime("%Y-%m-%d") # e.g., "2026-02-08"

# --- SYSTEM PROMPT (Dynamic Date Injection) ---
# --- SYSTEM PROMPT (Robust & Anti-Hallucination) ---
SYSTEM_PROMPT = f"""
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
  - "New York" $\rightarrow$ `JFK` or `EWR`
  - "Paris" $\rightarrow$ `CDG` or `ORY`
  - "London" $\rightarrow$ `LHR` or `LGW`
  - "Madurai" $\rightarrow$ `IXM`
  - "Chennai" $\rightarrow$ `MAA`
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
- **VALID `budgetLevel` VALUES:** the tool only accepts exactly `budget`, `mid-range`, or `luxury` (nothing else — e.g. NOT `"low"`, `"cheap"`). Map the user's wording: "cheap/low/minimum" → `budget`, "moderate/comfortable" → `mid-range`, "luxury/high-end" → `luxury`. If you send an invalid value, the tool silently falls back to `mid-range` pricing, which will be wrong.
- **IMPORTANT LIMITATION:** this tool does NOT accept real flight/hotel prices as input — it only returns a generic estimate for the chosen `budgetLevel`. Do NOT claim you "fed it the real price." Instead, report the tool's estimate labeled as "Generic estimate," and **separately** compute and clearly label a "Actual total (from real data found)" by summing the exact flight price + (hotel nightly price × nights) + a stated daily-expense estimate. When the user asks you to minimize cost, the actual total is what should change — the generic tool estimate will not.

### 🔁 FOLLOW-UP QUESTIONS (DO NOT RE-CALL TOOLS UNNECESSARILY)
- Prior tool results for this conversation are included below under `PRIOR TRIP DATA`, if any exist. Treat that as ground truth.
- If the user's follow-up can be answered by re-reasoning over `PRIOR TRIP DATA` (e.g. "pick the cheapest hotel from that list," "minimize cost," "which one is best for families") — DO NOT call any tool again. Just re-analyze the existing data and answer directly.
- Only call a tool again if the user asks for something the existing data cannot answer — a different city, different dates, a different category of place, or explicitly asks you to "search again" / "check for more/cheaper options."

### 📝 OUTPUT FORMAT
1. **Summary:** A quick breakdown of flight options and hotel recommendations.
2. **Itinerary:** A day-by-day plan using the specific *Attractions* found by `search_places`.
3. **Budget:** Both the generic tool estimate and the actual computed total (see above).
4. **Disclaimer:** "Prices and availability are subject to change."
"""

# --- SEMANTIC GATE (replaces keyword matching) ---
# A single fast, cheap model call that decides whether this turn needs new
# tool data or can be answered from what's already been fetched. Uses
# openai/gpt-oss-20b on Groq: ~1000 tok/s and priced for exactly this kind
# of high-frequency, low-output classification (single-word answer, so the
# added latency is small — typically well under the latency of even one
# real tool call, let alone four).
CLASSIFIER_MODEL = "openai/gpt-oss-20b"

async def needs_fresh_tool_data(latest_user_msg: str, trip_data: dict, last_assistant_msg: str) -> bool:
    """Returns True if this turn should have tools available, False if it
    can be answered purely by re-reasoning over trip_data. Fails open (True)
    on any error or ambiguous output, since an unnecessary tool call is far
    cheaper than a stranded, blank answer."""
    if not trip_data:
        return True  # nothing fetched yet — always allow the first search

    available = ", ".join(trip_data.keys())
    classifier_prompt = f"""You are a binary router for a travel-planning agent.

Data already fetched this session (tool names): {available}
Assistant's last answer (truncated): {last_assistant_msg[:500]}
User's new message: {latest_user_msg}

Can the user's new message be fully answered by re-analyzing the data already
fetched (e.g. picking a cheapest option, listing alternatives from a list
already returned, comparing items already found)? Or does it require fetching
NEW data — a different city/country, different dates, a different tool
category, or anything not already covered above?

Reply with exactly one word, nothing else: REUSE or FRESH."""

    try:
        classifier = ChatGroq(model=CLASSIFIER_MODEL, temperature=0, max_tokens=5)
        result = await classifier.ainvoke([HumanMessage(content=classifier_prompt)])
        verdict = (result.content or "").strip().upper()
        return not verdict.startswith("REUSE")
    except Exception:
        return True  # classifier failed — fail open, keep tools available


# --- CORE LOGIC ---
async def run_agent(chat_history, trip_data, chat_container):
    async with AsyncExitStack() as stack:
        status_text = chat_container.empty()
        status_text.info("🔌 Connecting to Server...")

        try:
            # 1. Connect
            transport = await stack.enter_async_context(sse_client(server_url))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            status_text.info("✅ Connected! Discovering tools...")

            # 2. List Tools
            mcp_tools = await session.list_tools()
            langchain_tools = []

            for tool in mcp_tools.tools:
                async def call_mcp_tool(tool_name=tool.name, **kwargs):
                    return await session.call_tool(tool_name, arguments=kwargs)

                input_model = create_pydantic_model_from_schema(tool.name, tool.inputSchema)
                lc_tool = StructuredTool.from_function(
                    func=None,
                    coroutine=call_mcp_tool,
                    name=tool.name,
                    description=tool.description,
                    args_schema=input_model
                )
                langchain_tools.append(lc_tool)
            
            status_text.info(f"🛠️ Found {len(langchain_tools)} tools. Thinking...")

            # 3. Initialize LLM
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

            # --- HARD GATE: decide in code, not just in the prompt, whether this turn
            # is even allowed to call tools. A system-prompt instruction like "don't
            # call tools on follow-ups" is only a suggestion the model can ignore.
            # The decision itself is semantic (via needs_fresh_tool_data), not a
            # keyword match, so it generalizes to phrasing we didn't anticipate
            # ("compare that to Singapore" correctly reads as needing fresh data).
            latest_user_msg = next(
                (m["content"] for m in reversed(chat_history) if m["role"] == "user"), ""
            )
            latest_assistant_msg = next(
                (m["content"] for m in reversed(chat_history) if m["role"] == "assistant"), ""
            )
            needs_fresh_search = await needs_fresh_tool_data(latest_user_msg, trip_data, latest_assistant_msg)

            llm_active = llm.bind_tools(langchain_tools) if needs_fresh_search else llm

            # 4. Construct Message History with System Prompt
            # Replay the full prior conversation (not just the latest turn) so the
            # model retains context across messages. Cap it to the last N turns to
            # keep token usage / latency bounded on long sessions.
            MAX_HISTORY_TURNS = 12
            trimmed_history = [m for m in chat_history if m["role"] in ("user", "assistant")][-MAX_HISTORY_TURNS:]

            messages = [SystemMessage(content=SYSTEM_PROMPT)]

            if trip_data:
                prior_data_block = "\n\n".join(
                    f"### {tool_name}\n{json.dumps(result)[:2000]}"
                    for tool_name, result in trip_data.items()
                )
                messages.append(SystemMessage(
                    content=f"PRIOR TRIP DATA (already fetched this session — reuse it, don't refetch unless truly needed):\n\n{prior_data_block}"
                ))

            for m in trimmed_history:
                if m["role"] == "user":
                    messages.append(HumanMessage(content=m["content"]))
                else:
                    messages.append(AIMessage(content=m["content"]))

            if not needs_fresh_search:
                status_text.info("💡 Answering from previously fetched trip data — no new search needed.")

            # 5. Agent Loop
            ai_msg = await llm_active.ainvoke(messages)
            messages.append(ai_msg)

            if ai_msg.tool_calls:
                status_text.info(f"🤔 Decided to call {len(ai_msg.tool_calls)} tools...")
                
                # Execute Tools
                for tool_call in ai_msg.tool_calls:
                    selected_tool = next((t for t in langchain_tools if t.name == tool_call['name']), None)
                    if selected_tool:
                        with st.chat_message("ai"):
                            st.write(f"🛠️ **Executing:** `{tool_call['name']}`")
                            st.json(tool_call['args'])
                        
                        # EXECUTE
                        tool_result = await selected_tool.coroutine(**tool_call['args'])
                        content_text = tool_result.content[0].text
                        
                        # --- DEBUG: CHECK FOR EMPTY DATA ---
                        if content_text == "[]" or content_text == "{}" or "error" in content_text.lower():
                             st.warning(f"⚠️ Tool {tool_call['name']} returned no data. Expect limited results.")

                        tool_msg = ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=content_text,
                            name=tool_call['name']
                        )
                        messages.append(tool_msg)

                        # Persist for future turns (keyed by tool name; last call wins)
                        try:
                            trip_data[tool_call['name']] = json.loads(content_text)
                        except (ValueError, TypeError):
                            trip_data[tool_call['name']] = content_text
            
            # 6. Final Answer
            status_text.info("📝 Generating final itinerary...")
            final_response = await llm_active.ainvoke(messages)
            status_text.empty()
            return final_response.content or (
                "I couldn't generate a proper answer for that — could you rephrase, "
                "or ask me to search fresh data for the new request?"
            )

        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            return None

# --- UI: Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "I am your AI Travel Agent. Where would you like to go?"}]

# Raw tool results from prior turns, keyed by tool name -> last result JSON.
# This is what lets the model answer follow-ups ("minimize cost", "pick the cheapest")
# without re-calling every tool from scratch each time.
if "trip_data" not in st.session_state:
    st.session_state.trip_data = {}

for message in st.session_state.messages:
    if message["role"] != "system": # Don't show system prompt in chat
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Where do you want to go?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = asyncio.run(run_agent(st.session_state.messages, st.session_state.trip_data, st.empty()))
        
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
