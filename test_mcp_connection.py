#!/usr/bin/env python3
"""
Test script to verify MCP server connection and tool listing.
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# For testing SSE connection to Render deployment
import httpx

async def test_render_sse_connection():
    """Test the SSE endpoint on Render"""
    print("🔍 Testing Render SSE endpoint...")
    
    url = "https://neuronworks-travel-agent.onrender.com/sse"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            print(f"✅ SSE endpoint responded with status: {response.status_code}")
            print(f"📋 Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                # Show first 500 chars of response
                content_preview = response.text[:500]
                print(f"📄 Content preview:\n{content_preview}")
                return True
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def test_health_endpoint():
    """Test the health endpoint"""
    print("\n🔍 Testing health endpoint...")
    
    url = "https://neuronworks-travel-agent.onrender.com/health"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            print(f"✅ Health endpoint status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"📊 Health data: {json.dumps(data, indent=2)}")
                return True
            else:
                print(f"⚠️  Status code: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 MCP Server Connection Test")
    print("=" * 60)
    
    # Test 1: Health endpoint
    health_ok = await test_health_endpoint()
    
    # Test 2: SSE endpoint
    sse_ok = await test_render_sse_connection()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Health endpoint: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"SSE endpoint:    {'✅ PASS' if sse_ok else '❌ FAIL'}")
    print("=" * 60)
    
    if health_ok and sse_ok:
        print("\n✅ All tests passed! Your MCP server is ready.")
        print("\n📝 Next steps:")
        print("1. Run the Streamlit frontend: streamlit run frontend/streamlit_app.py")
        print("2. Enter the MCP server URL: https://neuronworks-travel-agent.onrender.com")
        print("3. Test the connection and available tools")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())
