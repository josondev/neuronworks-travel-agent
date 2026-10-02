"""
Full MCP protocol test: connect via SSE, initialize, list tools.
"""
import httpx
import json
import threading
import time

BASE = "https://neuronworks-travel-agent.onrender.com"

session_id = None
message_endpoint = None
sse_messages = []

def listen_sse():
    """Listen to SSE stream in background thread."""
    global session_id, message_endpoint
    with httpx.stream("GET", f"{BASE}/sse", timeout=30) as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                data = line[5:].strip()
                sse_messages.append(data)
                if "/message?sessionId=" in data:
                    message_endpoint = data
                    session_id = data.split("sessionId=")[1]
                    print(f"  [SSE] Got session: {session_id[:16]}...")
                else:
                    try:
                        parsed = json.loads(data)
                        print(f"  [SSE] Response: {json.dumps(parsed)[:200]}")
                    except Exception:
                        pass

# Start SSE listener
print("1. Connecting to SSE...")
t = threading.Thread(target=listen_sse, daemon=True)
t.start()

# Wait for session ID
for _ in range(30):
    if session_id:
        break
    time.sleep(0.2)

if not session_id:
    print("ERROR: No session ID received from SSE stream")
    exit(1)

print(f"2. Session established: {session_id[:16]}...")
time.sleep(0.3)

# Send initialize
print("3. Sending initialize...")
init_payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0"}
    }
}
r = httpx.post(
    f"{BASE}/message?sessionId={session_id}",
    json=init_payload,
    timeout=10
)
print(f"  POST status: {r.status_code} | Body: {r.text[:100]}")
time.sleep(0.5)

# Send tools/list
print("4. Sending tools/list...")
tools_payload = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
}
r = httpx.post(
    f"{BASE}/message?sessionId={session_id}",
    json=tools_payload,
    timeout=10
)
print(f"  POST status: {r.status_code} | Body: {r.text[:100]}")
time.sleep(1.5)  # Wait for SSE response

# Print all tool names received
print()
print("=== Tools received via SSE ===")
for msg in sse_messages:
    try:
        parsed = json.loads(msg)
        if "result" in parsed and "tools" in parsed.get("result", {}):
            tools = parsed["result"]["tools"]
            print(f"Total tools: {len(tools)}")
            for tool in tools:
                print(f"  ✅ {tool['name']}: {tool.get('description','')[:60]}")
    except Exception:
        pass
