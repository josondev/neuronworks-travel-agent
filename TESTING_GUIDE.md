# Testing Guide - NeuronWorks Travel Agent

## 🚀 Deployment Status

### Latest Push
- **Commit**: `14f257f` - merge: resolve conflicts — keep local DB layer and bug fixes
- **Changes**: All bug fixes + complete PostgreSQL/Prisma 7 DB layer
- **GitHub**: https://github.com/josondev/neuronworks-travel-agent
- **Live URL**: https://neuronworks-travel-agent.onrender.com

### Wait for Render Deployment
Render auto-deploys on push. Wait ~2-5 minutes, then verify:

```bash
# Test 1: Check server is running
curl https://neuronworks-travel-agent.onrender.com/

# Test 2: Check health endpoint
curl https://neuronworks-travel-agent.onrender.com/health
```

Expected health response:
```json
{
  "status": "ok",
  "uptime": 123.45,
  "db": "connected"  // or "unavailable" if DATABASE_URL not set
}
```

---

## 🧪 Testing the MCP Server Connection

### Step 1: Start the Streamlit Frontend

```bash
cd c:\AI_PROJECTS\neuronworks-travel-agent
streamlit run frontend/streamlit_app.py
```

The app should open in your browser at `http://localhost:8501`

### Step 2: Connect to the MCP Server

In the Streamlit UI:
1. **Sidebar**: Find the "MCP Server URL" input field
2. **Enter**: `https://neuronworks-travel-agent.onrender.com`
3. **Click**: "Connect to MCP Server" button

### Step 3: Verify Connection

You should see:
- ✅ "Connected to MCP Server successfully!"
- A list of available tools displayed in the sidebar

**Expected Tools** (14 total):
1. `build_trip_data` - Full trip bundle orchestrator
2. `search_itineraries` - Search cached/live itineraries
3. `search_flights` - Flight search
4. `search_accommodations` - Hotel search
5. `search_places` - Places/attractions search
6. `search_restaurants` - Restaurant search
7. `get_weather` - Weather forecast
8. `calculate_budget` - Budget estimation
9. `convert_currency` - Currency conversion
10. `get_country_info` - Country information
11. **NEW** → `save_itinerary` - Save trip to database
12. **NEW** → `get_saved_itineraries` - Retrieve saved trips
13. **NEW** → `get_search_history` - View search history

---

## 🧪 Testing Database Features

### Test 1: Search Caching

1. **First Search**: Search for a trip (e.g., "New York to London, Jan 15-20")
   - Check console logs - should see cache MISS
   - Response time will be slower (hitting live APIs)

2. **Repeat Search**: Same exact search again
   - Should see cache HIT
   - Response time should be much faster (~100-200ms)

### Test 2: Save Itinerary

1. After getting search results, use the `save_itinerary` tool
2. Provide:
   - `userId`: any string (e.g., "test-user-1")
   - `tripData`: the complete trip JSON
   - `tripName`: descriptive name (e.g., "NYC to London - Jan 2027")

3. Should receive confirmation with itinerary ID

### Test 3: Retrieve Saved Itineraries

1. Use `get_saved_itineraries` tool
2. Provide the same `userId` from Test 2
3. Should see your saved trip(s) returned

### Test 4: Search History

1. Use `get_search_history` tool
2. Provide the same `userId`
3. Should see all your previous searches logged

---

## 🐛 Troubleshooting

### Issue: "Failed to connect to MCP server"

**Check:**
```bash
# 1. Is Render deployment live?
curl https://neuronworks-travel-agent.onrender.com/health

# 2. Is SSE endpoint accessible?
curl -N https://neuronworks-travel-agent.onrender.com/sse
```

**Solutions:**
- Wait for Render deployment to complete (~2-5 min after push)
- Check Render dashboard logs for errors
- Verify no CORS issues in browser console

### Issue: "db": "unavailable" in health check

This is expected if you haven't set up PostgreSQL yet. The app will use SQLite fallback.

**To add PostgreSQL:**
1. Go to Render Dashboard
2. Create a new PostgreSQL database
3. Copy the Internal Database URL
4. Add environment variable: `DATABASE_URL=<your-postgres-url>`
5. Redeploy

### Issue: Cache not working

**Verify:**
- Check if `search_cache` table exists (SQLite: `prisma/dev.db`)
- Look for console logs showing cache hits/misses
- Ensure Prisma migrations ran: `npx prisma migrate deploy`

### Issue: Tools not showing in frontend

**Check:**
1. MCP server version matches: Node SDK `1.26.0`, Python `mcp==1.1.2`
2. SSE connection established (check browser network tab)
3. No errors in Streamlit console

---

## 📊 Expected Results

### Successful Connection
```
✅ Server responds at /health
✅ SSE endpoint streams connection-open events
✅ Tool list returns 14 tools (including 3 new DB tools)
✅ First search hits live APIs (slower)
✅ Repeated search hits cache (much faster)
✅ Saved itineraries persist across sessions
✅ Search history tracks all queries
```

### Performance Benchmarks
- **Cache hit**: ~100-200ms
- **Cache miss** (live API): ~2-5 seconds
- **SSE connection**: <1 second
- **Tool listing**: <500ms

---

## 🔍 Debugging Tips

### View Render Logs
```bash
# Install Render CLI (if needed)
npm install -g render-cli

# Login and view logs
render login
render logs -s <your-service-name>
```

### Local Testing (Bypass Render)
```bash
# Run server locally
npm run dev

# Connect frontend to localhost
# In Streamlit: http://localhost:3000 (or whatever port npm run dev uses)
```

### Verify Prisma Schema
```bash
# Check database connection
npx prisma db push

# View data in Prisma Studio
npx prisma studio
```

---

## ✅ Success Criteria

Your deployment is fully working when:

1. ✅ `/health` endpoint returns `200 OK` with uptime
2. ✅ Streamlit frontend connects successfully
3. ✅ All 14 tools are listed
4. ✅ `build_trip_data` returns flight/hotel/weather data
5. ✅ Repeated searches return faster (cache working)
6. ✅ `save_itinerary` persists data
7. ✅ `get_saved_itineraries` retrieves saved trips
8. ✅ `get_search_history` shows query log

---

## 📝 Next Steps After Successful Test

1. **Add PostgreSQL** for production persistence (SQLite is ephemeral on Render)
2. **Monitor cache hit rate** - tune similarity threshold if needed
3. **Add user authentication** for multi-user support
4. **Implement cache pruning** job to clean old entries
5. **Add analytics** to track most popular destinations

Good luck! 🚀
