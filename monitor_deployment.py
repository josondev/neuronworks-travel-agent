#!/usr/bin/env python3
"""
Monitor Render deployment status until ready.
"""
import asyncio
import httpx
from datetime import datetime

async def check_deployment():
    """Check if deployment is ready"""
    url = "https://neuronworks-travel-agent.onrender.com/health"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return True, data
            else:
                return False, f"Status: {response.status_code}"
                
    except Exception as e:
        return False, str(e)

async def monitor():
    """Monitor deployment with status updates"""
    print("=" * 60)
    print("🔍 Monitoring Render Deployment")
    print("=" * 60)
    print("Checking every 15 seconds...")
    print("Press Ctrl+C to stop\n")
    
    attempt = 0
    
    while True:
        attempt += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        ready, result = await check_deployment()
        
        if ready:
            print(f"\n{timestamp} - ✅ DEPLOYMENT READY!")
            print(f"Health data: {result}")
            print("\n" + "=" * 60)
            print("🚀 Ready to test!")
            print("=" * 60)
            print("\nRun: streamlit run frontend/streamlit_app.py")
            print("Then connect to: https://neuronworks-travel-agent.onrender.com")
            break
        else:
            status_symbol = "⏳" if attempt % 2 == 0 else "⏱️"
            print(f"{timestamp} - {status_symbol} Attempt {attempt}: Not ready yet ({result})")
        
        await asyncio.sleep(15)

if __name__ == "__main__":
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
