import streamlit as st
import asyncio
import os
import nest_asyncio
from contextlib import AsyncExitStack
from pydantic import create_model, Field

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

# 1. Apply Async Patch (Crucial for Streamlit)
nest_asyncio.apply()

# 2. Page Config
st.set_page_config(page_title="AI Travel Agent", page_icon="✈️")
st.title("✈️ AI Travel Agent")
st.caption("Powered by NeuralWorks & MCP")

# 3. Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    server_url = st.text_input("MCP Server URL", value="https://neuronworks-travel-agent.onrender.com/sse")
    
    # Get API Key from secrets or user input
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

# --- CORE LOGIC: The Agent Workflow ---
async def run_agent(query, chat_container):
    async with AsyncExitStack() as stack:
        # Status Update
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
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)
            llm_with_tools = llm.bind_tools(langchain_tools)
            
            messages = [HumanMessage(content=query)]

            # 4. Agent Loop
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
                        
                        tool_result = await selected_tool.coroutine(**tool_call['args'])
                        content_text = tool_result.content[0].text
                        
                        tool_msg = ToolMessage(
                            tool_call_id=tool_call['id'],
                            content=content_text,
                            name=tool_call['name']
                        )
                        messages.append(tool_msg)
            
            # 5. Final Answer
            status_text.info("📝 Generating final itinerary...")
            final_response = await llm_with_tools.ainvoke(messages)
            status_text.empty() # Clear status
            return final_response.content

        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            return None

# --- UI: Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("Where do you want to go?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Run the async agent loop
        response = asyncio.run(run_agent(prompt, st.empty()))
        
        if response:
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
