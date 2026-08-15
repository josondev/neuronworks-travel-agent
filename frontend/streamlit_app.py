import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import nest_asyncio
import streamlit as st
from pydantic import Field, create_model
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

nest_asyncio.apply()
st.set_page_config(page_title="Neuronworks Travel Agent", page_icon="✈️", layout="wide")

# ----------------------------- UI ---------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 10% 0%, #1d4ed8 0, transparent 35%),
                radial-gradient(circle at 90% 10%, #7c3aed 0, transparent 32%),
                #080b14;
}
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 7rem; }
.hero {
    padding: 28px 30px; border-radius: 24px; margin-bottom: 22px;
    background: linear-gradient(135deg, rgba(37,99,235,.30), rgba(124,58,237,.25));
    border: 1px solid rgba(255,255,255,.12);
    box-shadow: 0 18px 60px rgba(0,0,0,.25);
}
.hero h1 { margin: 0; color: #fff; font-size: 2.25rem; letter-spacing: -.04em; }
.hero p { margin: 8px 0 0; color: #cbd5e1; font-size: 1rem; }
.status-pill {
    display:inline-block; padding:5px 11px; border-radius:999px;
    background:rgba(255,255,255,.10); color:#e2e8f0; font-size:.78rem;
    border:1px solid rgba(255,255,255,.12);
}
section[data-testid="stSidebar"] { background: rgba(8,11,20,.96); border-right: 1px solid rgba(255,255,255,.08); }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color:#f8fafc; }
div[data-testid="stChatMessage"] {
    border: 1px solid rgba(255,255,255,.09); border-radius: 20px;
    padding: 1.15rem 1.25rem; margin: .8rem 0;
    background: rgba(15,23,42,.76); box-shadow: 0 10px 30px rgba(0,0,0,.18);
}
div[data-testid="stChatMessageContent"] { color:#e5e7eb; }
div[data-testid="stChatMessageContent"] h1,
div[data-testid="stChatMessageContent"] h2,
div[data-testid="stChatMessageContent"] h3 { color:#fff; margin-top:.5rem; }
div[data-testid="stChatMessageContent"] table { border-radius:12px; overflow:hidden; }
div[data-testid="stChatMessageContent"] code { color:#c4b5fd; }
div[data-testid="stExpander"] { border:1px solid rgba(255,255,255,.09); border-radius:14px; }
[data-testid="stChatInput"] { border-radius:18px; }
.small-muted { color:#94a3b8; font-size:.82rem; }
.metric-card { padding:14px 16px; border-radius:16px; background:rgba(15,23,42,.8); border:1px solid rgba(255,255,255,.08); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="status-pill">● LIVE MCP TRAVEL INTELLIGENCE</div>
  <h1>✈️ Neuronworks Travel Agent</h1>
  <p>Flights · Hotels · Places · Restaurants · Weather · Currency · Budget</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    server_url = st.text_input("MCP Server URL", value="https://neuronworks-travel-agent.onrender.com/sse")

    groq_api_key = os.environ.get("GROQ_API_KEY") or st.text_input("Groq API Key", type="password")
    hf_token = os.environ.get("HF_TOKEN") or st.text_input("Hugging Face Token", type="password")

    if not groq_api_key:
        st.warning("Enter GROQ_API_KEY for the GPT-OSS semantic router.")
        st.stop()
    if not hf_token:
        st.warning("Enter HF_TOKEN for Llama 3.3 70B.")
        st.stop()

    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["HF_TOKEN"] = hf_token
    st.success("🟢 Connected")
    st.markdown('<div class="small-muted">Main planner: Meta Llama 3.3 70B Instruct · Router: GPT-OSS 20B</div>', unsafe_allow_html=True)

# -------------------------- Schema helpers -------------------------
def schema_type(schema: Dict[str, Any]):
    if not isinstance(schema, dict):
        return Any
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        try:
            return Literal[tuple(schema["enum"])]
        except TypeError:
            pass
    t = schema.get("type")
    if t == "string": return str
    if t == "number": return float
    if t == "integer": return int
    if t == "boolean": return bool
    if t == "array": return List[schema_type(schema.get("items", {}))]
    if t == "object": return Dict[str, Any]
    any_of = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(any_of, list):
        types = [schema_type(x) for x in any_of if x.get("type") != "null"]
        if len(types) == 1: return Optional[types[0]]
        if types: return Union[tuple(types)]
    return Any


def create_pydantic_model_from_schema(name: str, schema: Dict[str, Any]):
    fields = {}
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    for field_name, info in properties.items():
        annotation = schema_type(info)
        default = info.get("default", ... if field_name in required else None)
        if field_name not in required and default is None:
            annotation = Optional[annotation]
        fields[field_name] = (annotation, Field(default=default, description=info.get("description", "")))
    return create_model(f"{name}Input", **fields)

# -------------------------- Agent ----------------------------------
current_date = datetime.now().strftime("%Y-%m-%d")

SYSTEM_PROMPT = f"""
You are Neuronworks Travel Agent, a factual travel-planning assistant.
TODAY: {current_date}

NON-NEGOTIABLE:
- Never invent flight details, hotel prices, ratings, attractions, weather, exchange rates, or availability.
- If the user did not provide enough information to search a trip (especially dates/origin/destination), ask a concise clarification instead of assuming.
- For a COMPLETE FIRST TRIP PLAN, use the MCP tool `build_trip_data`. It is the orchestration tool and MUST be preferred over separately calling search_flights/search_hotels/search_places/weather/budget. It runs the complete service layer together.
- `build_trip_data` returns flights, hotels, attractions, restaurants, weather, currency when requested, and a generic budget estimate.
- Flight origin/destination must be IATA codes. Common mappings: Chennai MAA, Madurai IXM, Colombo CMB, London LHR, Paris CDG, New York JFK/EWR.
- Never describe the generic budget estimate as the actual booking total.
- If a service fails, explicitly label that service as unavailable; do not fill the gap with guesses.
- A return date must be after the departure date.
- For follow-ups that only compare or re-rank already returned data, reuse the existing trip data.
- If the destination, dates, origin, or traveler count changes, fetch fresh data.

OUTPUT RULES — VERY IMPORTANT:
- Produce ONE final answer only. Never write "Step 1..." followed by another "final answer".
- Never expose your internal reasoning or tool-selection process.
- Do not repeat the same itinerary twice.
- Make the answer visually clean and compact using Markdown headings, tables, bullets, and emojis sparingly.
- Prefer this structure:
  ## ✈️ Trip at a glance
  ## 🛫 Flights
  ## 🏨 Hotels
  ## 📍 Things to do
  ## 🍽️ Food picks
  ## 🌦️ Weather
  ## 💰 Budget
  ## 🗓️ Suggested itinerary
  ## ⚠️ Notes
- Only include sections for data that actually exists.
- If exact live prices exist, show them exactly as returned.
- If a live service has no result, say so plainly.
"""

CLASSIFIER_MODEL = "openai/gpt-oss-20b"

async def needs_fresh_tool_data(user_msg: str, trip_data: dict, last_answer: str) -> bool:
    if not trip_data:
        return True
    prompt = f"""Classify this travel follow-up as REUSE or FRESH.
Existing data: {', '.join(trip_data.keys())}
Last answer: {last_answer[:500]}
User: {user_msg}
REUSE only if the request can be fully answered from existing data. FRESH if it needs a new destination, dates, origin, travelers, category, or live search.
Reply with exactly REUSE or FRESH."""
    try:
        router = ChatGroq(model=CLASSIFIER_MODEL, temperature=0, max_tokens=5)
        result = await router.ainvoke([HumanMessage(content=prompt)])
        return not (result.content or "").strip().upper().startswith("REUSE")
    except Exception:
        return True


def compact_json(value, limit=8000):
    try:
        return json.dumps(value, ensure_ascii=False)[:limit]
    except Exception:
        return str(value)[:limit]


def infer_bundle_args(tool_calls):
    """Turn the model's accidental individual first-plan calls into one bundle call.
    This is the safety net that guarantees the complete service layer is used even if
    Llama chooses search_flights/search_hotels/search_places individually.
    """
    args = {}
    for call in tool_calls:
        name = call.get("name")
        a = call.get("args") or {}
        if name == "search_flights":
            args.update({
                "origin": a.get("origin"),
                "destinationAirport": a.get("destination"),
                "departDate": a.get("departDate"),
                "returnDate": a.get("returnDate"),
                "passengers": a.get("passengers", 1)
            })
        elif name == "search_hotels":
            args.setdefault("destinationCity", a.get("city"))
            args.setdefault("departDate", a.get("checkIn"))
            args.setdefault("returnDate", a.get("checkOut"))
            args.setdefault("passengers", a.get("adults", 1))
        elif name == "search_places":
            args.setdefault("destinationCity", a.get("location"))
    if args.get("destinationCity") and args.get("origin") and args.get("destinationAirport") and args.get("departDate") and args.get("returnDate"):
        args["destinationCountry"] = "Unknown"
        args["budgetLevel"] = "budget"
        return args
    return None


async def run_agent(chat_history, trip_data, chat_container):
    async with AsyncExitStack() as stack:
        status = chat_container.empty()
        status.info("🔌 Connecting to MCP server…")
        try:
            transport = await stack.enter_async_context(sse_client(server_url))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            status.info("🧰 Loading travel services…")

            mcp_tools = await session.list_tools()
            tools = []
            for tool in mcp_tools.tools:
                async def call_mcp_tool(tool_name=tool.name, **kwargs):
                    return await session.call_tool(tool_name, arguments=kwargs)
                tools.append(StructuredTool.from_function(
                    func=None,
                    coroutine=call_mcp_tool,
                    name=tool.name,
                    description=tool.description or "Travel MCP tool",
                    args_schema=create_pydantic_model_from_schema(tool.name, tool.input_schema)
                ))

            endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                task="text-generation",
                max_new_tokens=1800,
                temperature=0.01,
                huggingfacehub_api_token=os.environ["HF_TOKEN"]
            )
            llm = ChatHuggingFace(llm=endpoint)
            llm_tools = llm.bind_tools(tools)

            latest_user = next((m["content"] for m in reversed(chat_history) if m["role"] == "user"), "")
            last_answer = next((m["content"] for m in reversed(chat_history) if m["role"] == "assistant"), "")
            fresh = await needs_fresh_tool_data(latest_user, trip_data, last_answer)

            if fresh and trip_data:
                trip_data.clear()

            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            if trip_data:
                messages.append(SystemMessage(content="CURRENT TRIP DATA (ground truth):\n" + "\n\n".join(
                    f"### {k}\n{compact_json(v, 5000)}" for k, v in trip_data.items()
                )))

            for m in [x for x in chat_history if x["role"] in ("user", "assistant")][-12:]:
                messages.append(HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))

            if fresh:
                status.info("🧠 Planning with live travel services…")
                first = await llm_tools.ainvoke(messages)

                # If Llama chose individual search tools, convert that choice into the
                # complete orchestration tool so every service is actually exercised.
                bundle = infer_bundle_args(first.tool_calls or [])
                bundle_tool = next((t for t in tools if t.name == "build_trip_data"), None)

                if bundle and bundle_tool:
                    with st.expander("🧰 Live services used", expanded=False):
                        st.caption("Flights · Hotels · Attractions · Restaurants · Weather · Budget")
                        st.json(bundle)
                    status.info("⚡ Running the complete service bundle…")
                    result = await bundle_tool.coroutine(**bundle)
                    content = result.content[0].text
                    messages.append(AIMessage(content="I requested the complete live trip bundle."))
                    messages.append(ToolMessage(tool_call_id="bundle", name="build_trip_data", content=content))
                    try:
                        trip_data["build_trip_data"] = json.loads(content)
                    except Exception:
                        trip_data["build_trip_data"] = content
                else:
                    # Genuine follow-up/new-data request where a bundle cannot be formed.
                    messages.append(first)
                    for call in first.tool_calls or []:
                        selected = next((t for t in tools if t.name == call["name"]), None)
                        if not selected:
                            continue
                        with st.expander(f"🧰 {call['name']}", expanded=False):
                            st.json(call["args"])
                        try:
                            result = await selected.coroutine(**call["args"])
                            content = result.content[0].text
                        except Exception as exc:
                            content = json.dumps({"error": str(exc)})
                        messages.append(ToolMessage(tool_call_id=call["id"], name=call["name"], content=content))
                        try: trip_data[call["name"]] = json.loads(content)
                        except Exception: trip_data[call["name"]] = content
            else:
                status.info("♻️ Reusing your existing live trip data…")

            status.info("✨ Writing your itinerary…")
            final = await llm.ainvoke(messages)
            status.empty()
            return final.content or "I couldn't generate a response from the available live data."

        except Exception as exc:
            status.error(f"Something went wrong: {exc}")
            return None

# -------------------------- Chat -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "trip_data" not in st.session_state:
    st.session_state.trip_data = {}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Where are you going? Try: Chennai → Madurai, Aug 20–25, 2 people")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = asyncio.run(run_agent(st.session_state.messages, st.session_state.trip_data, st.empty()))
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
