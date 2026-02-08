import streamlit as st
import asyncio
import os
import nest_asyncio
from contextlib import AsyncExitStack
from pydantic import create_model, Field
from datetime import datetime 

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool

# 1. Apply Async Patch
nest_asyncio.apply()

# 2. Page Config
st.set_page_config(page_title="AI Travel Agent", page_icon="✈️")
st.title("✈️ AI Travel Agent")
st.caption("Powered by NeuralWorks & MCP")

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
- **EXECUTION:** Call this tool **LAST**.
- **DATA SOURCE:** Feed the *actual* flight price and *actual* (or estimated) hotel price you found into this tool. Do not use the default values if you have real data.

### 📝 OUTPUT FORMAT
1. **Summary:** A quick breakdown of flight options and hotel recommendations.
2. **Itinerary:** A day-by-day plan using the specific *Attractions* found by `search_places`.
3. **Budget:** A total cost estimation.
4. **Disclaimer:** "Prices and availability are subject to change."
"""

# --- CORE LOGIC ---
async def run_agent(query, chat_container):
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
            llm_with_tools = llm.bind_tools(langchain_tools)
            
            # 4. Construct Message History with System Prompt
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query)
            ]

            # 5. Agent Loop
            ai_msg = await llm_with_tools.ainvoke(messages)
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
            
            # 6. Final Answer
            status_text.info("📝 Generating final itinerary...")
            final_response = await llm_with_tools.ainvoke(messages)
            status_text.empty() 
            return final_response.content

        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            return None

# --- UI: Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "I am your AI Travel Agent. Where would you like to go?"}]

for message in st.session_state.messages:
    if message["role"] != "system": # Don't show system prompt in chat
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Where do you want to go?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = asyncio.run(run_agent(prompt, st.empty()))
        
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
