import httpx
import json

BASE = "https://neuronworks-travel-agent.onrender.com"

print("=== Test 1: Root ===")
r = httpx.get(f"{BASE}/", timeout=10)
print(f"Status: {r.status_code} | Body: {r.text}")

print()
print("=== Test 2: Health ===")
r = httpx.get(f"{BASE}/health", timeout=10)
print(f"Status: {r.status_code}")
data = r.json()
print(json.dumps(data, indent=2))

print()
print("=== Test 3: SSE stream (first 6 lines) ===")
with httpx.stream("GET", f"{BASE}/sse", timeout=15) as r:
    print(f"Status: {r.status_code}")
    ct = r.headers.get("content-type", "N/A")
    print(f"Content-Type: {ct}")
    count = 0
    for line in r.iter_lines():
        print(f"  >> {repr(line)}")
        count += 1
        if count >= 6:
            break

print()
print("=== All tests done ===")
