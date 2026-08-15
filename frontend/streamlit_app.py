import streamlit as st
import asyncio
import os
import json
import nest_asyncio
from contextlib import AsyncExitStack
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import create_model, Field
from mcp.client.sse import sse_client
from mcp import ClientSession
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool

st.set_page_config(page_title="AI Travel Agent", page_icon="✈️")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
section.main > div { max-width: 100%; padding: 1rem; }
div[data-testid="stChatMessage"] {
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.12);
}
@media (prefers-color-scheme: dark) {
    div[data-testid="stChatMessage"] { background: #0b0b0b !important; }
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] div { color: #f7fafc !important; }
}
@media (prefers-color-scheme: light) {
    div[data-testid="stChatMessage"] { background: #ffffff !important; }
    div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessageContent"] p,
    div[data-testid="stChatMessageContent"] li,
    div[data-testid="stChatMessageContent"] span,
    div[data-testid="stChatMessageContent"] div { color: #111827 !important; }
}
</style>
""", unsafe_allow_html=True)

nest_asyncio.apply()
st.title("✈️ AI Travel Agent")
st.caption("Powered by Neuronworks.ai & MCP")

with st.sidebar:
    st.header("Configuration")
    server_url = st.text_input(
        "MCP Server URL",
        value="https://neuronworks-travel-agent.onrender.com/sse"
    )

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password")
    if not groq_api_key:
        st.warning("⚠️ Please enter a Groq API Key for the GPT-OSS classifier.")
        st.stop()
    os.environ["GROQ_API_KEY"] = groq_api_key

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        hf_token = st.text_input("Hugging Face Token", type="password")
    if not hf_token:
        st.warning("⚠️ Please enter a Hugging Face Token for Llama 3.3 70B.")
        st.stop()
    os.environ["HF_TOKEN"] = hf_token

    st.success("✅ Ready to fly!")


def schema_type(field_schema: Dict[str, Any]):
    """Convert the MCP JSON Schema type into a useful Python/Pydantic type."""
    if not isinstance(field_schema, dict):
        return Any

    if "enum" in field_schema and isinstance(field_schema["enum"], list):
        values = tuple(field_schema["enum"])
        if values:
            try:
                return Literal[values]
            except TypeError:
                pass

    field_type = field_schema.get("type")

    if field_type == "string":
        return str
    if field_type == "number":
        return float
    if field_type == "integer":
        return int
    if field_type == "boolean":
        return bool
    if field_type == "array":
        return List[schema_type(field_schema.get("items", {}))]
    if field_type == "object":
        return Dict[str, Any]

    # Handle common JSON-schema unions such as nullable fields.
    any_of = field_schema.get("anyOf") or field_schema.get("oneOf")
    if isinstance(any_of, list):
        types = [schema_type(item) for item in any_of if item.get("type") != "null"]
        if len(types) == 1:
            return Optional[types[0]]
        if types:
            return Union[tuple(types)]

    return Any


def create_pydantic_model_from_schema(name: str, schema: Dict[str, Any]):
    fields = {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()

    for field_name, field_info in properties.items():
        annotation = schema_type(field_info)
        description = field_info.get("description", "")
        default = field_info.get("default", ... if field_name in required else None)

        if field_name not in required and default is None:
            annotation = Optional[annotation]

        fields[field_name] = (
            annotation,
            Field(default=default, description=description)
        )

    return create_model(f"{name}Input", **fields)


current_date = datetime.now().strftime("%Y-%m-%d")

SYSTEM_PROMPT = f"""
You are an expert, factual AI Travel Agent. Plan realistic trips using ONLY real-time
information returned by the MCP tools. Never invent prices, availability, weather,
exchange rates, hotel ratings, attractions, booking links, or flight details.

TODAY'S DATE: {current_date}

CRITICAL RULES
1. If a required date, origin, destination, or traveler count is genuinely missing,
   ask the user instead of inventing it. Never assume "tomorrow" unless the user says so.
2. Flight searches require 3-letter IATA airport codes and YYYY-MM-DD dates.
   Chennai=MAA, Madurai=IXM, London=LHR/LGW, Paris=CDG/ORY, New York=JFK/EWR.
3. Hotel searches require city + check-in + check-out dates + adults.
4. Budget is a GENERIC ESTIMATE only. It does not contain live flight/hotel prices.
5. Actual totals may only use prices explicitly returned by live tools. If a component
   is unavailable, say so instead of estimating a specific number.
6. If a tool returns an error or an empty result, clearly say that live data was unavailable.
7. Do not turn a generic estimate into a real quote.
8. Keep currencies exactly as returned unless the user explicitly asks for conversion.
9. For a complete first-trip plan, use this logical order where applicable:
   flights -> hotels -> places -> weather/currency -> budget LAST -> final answer.
10. For follow-ups that only compare/re-rank already fetched results, reuse prior data and
    do not call tools unnecessarily.
11. Prior trip data is valid ONLY for the same trip. If the user changes destination, dates,
    origin, or traveler count, use fresh data.

SUPPORTED PLACE CATEGORIES:
tourist_attractions, restaurants, entertainment, nature, shopping, religion

OUTPUT FORMAT
- Summary
- Flight options (if searched)
- Hotel options (if searched)
- Day-by-day itinerary using actual places returned by tools
- Budget: generic estimate + clearly labelled live-data subtotal only when defensible
- Disclaimer: Prices and availability are subject to change.
"""

CLASSIFIER_MODEL = "openai/gpt-oss-20b"


async def needs_fresh_tool_data(latest_user_msg: str, trip_data: dict, last_assistant_msg: str) -> bool:
    if not trip_data:
        return True

    available = ", ".join(trip_data.keys())
    classifier_prompt = f"""You are a binary router for a travel-planning agent.

Existing tool data: {available}
Last assistant answer: {last_assistant_msg[:600]}
New user message: {latest_user_msg}

Return REUSE if the new request can be answered completely from the existing data.
Return FRESH if it asks for a different destination, dates, origin, travelers, a new
category/tool, more options, or any live information not already present.
Reply with exactly one word: REUSE or FRESH."""

    try:
        classifier = ChatGroq(
            model=CLASSIFIER_MODEL,
            temperature=0,
            max_tokens=5
        )
        result = await classifier.ainvoke([HumanMessage(content=classifier_prompt)])
        verdict = (result.content or "").strip().upper()
        return not verdict.startswith("REUSE")
    except Exception:
        return True


async def run_agent(chat_history, trip_data, chat_container):
    async with AsyncExitStack() as stack:
        status_text = chat_container.empty()
        status_text.info("🔌 Connecting to Server...")

        try:
            transport = await stack.enter_async_context(sse_client(server_url))
            session = await stack.enter_async_context(
                ClientSession(transport[0], transport[1])
            )
            status_text.info("✅ Connected! Discovering tools...")

            mcp_tools = await session.list_tools()
            langchain_tools = []

            for tool in mcp_tools.tools:
                async def call_mcp_tool(tool_name=tool.name, **kwargs):
                    return await session.call_tool(tool_name, arguments=kwargs)

                input_model = create_pydantic_model_from_schema(
                    tool.name, tool.input_schema
                )
                langchain_tools.append(
                    StructuredTool.from_function(
                        func=None,
                        coroutine=call_mcp_tool,
                        name=tool.name,
                        description=tool.description or "MCP travel tool",
                        args_schema=input_model
                    )
                )

            status_text.info(f"🛠️ Found {len(langchain_tools)} tools. Thinking...")

            # Main agent: Hugging Face Llama 3.3 70B Instruct.
            # GPT-OSS 20B remains on Groq only for the cheap semantic router.
            hf_endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.3-70B-Instruct",
                task="text-generation",
                max_new_tokens=2048,
                temperature=0.01,
                huggingfacehub_api_token=os.environ["HF_TOKEN"]
            )
            llm = ChatHuggingFace(llm=hf_endpoint)

            latest_user_msg = next(
                (m["content"] for m in reversed(chat_history) if m["role"] == "user"),
                ""
            )
            latest_assistant_msg = next(
                (m["content"] for m in reversed(chat_history) if m["role"] == "assistant"),
                ""
            )

            needs_fresh_search = await needs_fresh_tool_data(
                latest_user_msg,
                trip_data,
                latest_assistant_msg
            )

            # A fresh request starts a fresh trip-data context. This prevents old
            # results such as Madurai from contaminating a new Sri Lanka request.
            if needs_fresh_search and trip_data:
                trip_data.clear()

            llm_with_tools = llm.bind_tools(langchain_tools)
            llm_active = llm_with_tools if needs_fresh_search else llm

            MAX_HISTORY_MESSAGES = 12
            trimmed_history = [
                m for m in chat_history
                if m["role"] in ("user", "assistant")
            ][-MAX_HISTORY_MESSAGES:]

            messages = [SystemMessage(content=SYSTEM_PROMPT)]

            if trip_data:
                prior_data_block = "\n\n".join(
                    f"### {tool_name}\n{json.dumps(result)[:3000]}"
                    for tool_name, result in trip_data.items()
                )
                messages.append(SystemMessage(
                    content=(
                        "PRIOR TRIP DATA — use this as ground truth for the current trip "
                        "and do not refetch unless the user requests new information:\n\n"
                        + prior_data_block
                    )
                ))

            for message in trimmed_history:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                else:
                    messages.append(AIMessage(content=message["content"]))

            if not needs_fresh_search:
                status_text.info("💡 Reusing previously fetched trip data — no new search needed.")

            # Tool phase. Multiple rounds let the model react to tool results.
            if needs_fresh_search:
                for round_number in range(3):
                    ai_msg = await llm_with_tools.ainvoke(messages)
                    messages.append(ai_msg)

                    if not ai_msg.tool_calls:
                        break

                    status_text.info(
                        f"🤔 Tool round {round_number + 1}: executing {len(ai_msg.tool_calls)} tool(s)..."
                    )

                    priority = {
                        "search_flights": 10,
                        "search_hotels": 20,
                        "search_places": 30,
                        "get_weather_forecast": 40,
                        "get_exchange_rate": 50,
                        "calculate_trip_budget": 100
                    }
                    ordered_calls = sorted(
                        ai_msg.tool_calls,
                        key=lambda call: priority.get(call["name"], 60)
                    )

                    for tool_call in ordered_calls:
                        selected_tool = next(
                            (tool for tool in langchain_tools if tool.name == tool_call["name"]),
                            None
                        )

                        if not selected_tool:
                            continue

                        with st.chat_message("ai"):
                            st.write(f"🛠️ **Executing:** `{tool_call['name']}`")
                            st.json(tool_call["args"])

                        try:
                            tool_result = await selected_tool.coroutine(**tool_call["args"])
                            content_text = tool_result.content[0].text
                        except Exception as tool_error:
                            content_text = json.dumps({
                                "error": f"Tool execution failed: {tool_error}"
                            })

                        if not content_text or content_text in ("[]", "{}") or '"error"' in content_text.lower():
                            st.warning(
                                f"⚠️ `{tool_call['name']}` returned no usable live data."
                            )

                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content=content_text,
                                name=tool_call["name"]
                            )
                        )

                        try:
                            trip_data[tool_call["name"]] = json.loads(content_text)
                        except (ValueError, TypeError):
                            trip_data[tool_call["name"]] = content_text

                    # If the model emitted tool calls, the next round can use their results.
                    # Budget is deliberately handled after other tool calls by priority.

            # IMPORTANT: final answer is generated WITHOUT tools bound.
            # This prevents the model from trying to call another tool instead of answering.
            status_text.info("📝 Generating final itinerary...")
            final_response = await llm.ainvoke(messages)
            status_text.empty()

            return final_response.content or (
                "I couldn't generate a proper answer from the available data. "
                "Please try the request again."
            )

        except Exception as error:
            status_text.error(f"Error: {error}")
            return None


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "I am your AI Travel Agent. Where would you like to go?"
        }
    ]

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
        response = asyncio.run(
            run_agent(
                st.session_state.messages,
                st.session_state.trip_data,
                st.empty()
            )
        )

        if response:
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
