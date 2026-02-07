#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
// 1. SILENCE LOGS
const originalLog = console.log;
console.log = console.error;
dotenv.config();
import { FlightService } from './services/FlightService.js';
import { AccommodationService } from './services/AccommodationService.js';
import { CurrencyService } from './services/CurrencyService.js';
import { WeatherService } from './services/WeatherService.js';
import { PlacesService } from './services/PlacesService.js';
const app = express();
app.use(cors({ origin: '*', methods: ['GET', 'POST', 'OPTIONS'] }));
app.use(express.json());
const flightService = new FlightService();
const accommodationService = new AccommodationService();
const currencyService = new CurrencyService();
const weatherService = new WeatherService();
const placesService = new PlacesService();
// Store active sessions
const sessions = new Map();
app.get('/', (req, res) => {
    res.status(200).send('Travel MCP Server is Running (Monkey Patch Mode)');
});
// --- SSE ENDPOINT ---
app.get('/sse', async (req, res) => {
    console.error('🔗 NEW CONNECTION: Client connected via SSE');
    // Write headers with correct SSE format
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache, must-revalidate');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('X-Accel-Buffering', 'no');
    // DO NOT call writeHead - let the SDK handle it!
    // DO NOT write before connecting - let SDK send headers first!
    // Create transport WITHOUT interfering with writeHead
    const transport = new SSEServerTransport('/message', res);
    const server = new Server({ name: 'travel-planner-server', version: '0.1.0' }, { capabilities: { tools: {} } });
    // --- TOOL DEFINITIONS ---
    server.setRequestHandler(ListToolsRequestSchema, async () => {
        console.error('📋 Client requested Tool List');
        return {
            tools: [
                {
                    name: 'search_flights',
                    description: 'Search for flight prices',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            origin: { type: 'string' },
                            destination: { type: 'string' },
                            departDate: { type: 'string' },
                            returnDate: { type: 'string' },
                            passengers: { type: 'number', default: 1 }
                        },
                        required: ['origin', 'destination', 'departDate']
                    }
                },
                {
                    name: 'get_weather_forecast',
                    description: 'Get weather forecast',
                    inputSchema: {
                        type: 'object',
                        properties: { city: { type: 'string' }, country: { type: 'string' }, startDate: { type: 'string' }, endDate: { type: 'string' } },
                        required: ['city', 'country', 'startDate', 'endDate']
                    }
                },
                {
                    name: 'calculate_trip_budget',
                    description: 'Calculate trip budget',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            destinations: { type: 'array', items: { type: 'string' } },
                            duration: { type: 'number' },
                            travelers: { type: 'number', default: 1 },
                            budgetLevel: { type: 'string', enum: ['budget', 'mid-range', 'luxury'] }
                        },
                        required: ['destinations', 'duration', 'budgetLevel']
                    }
                },
                {
                    name: 'search_places',
                    description: 'Search for tourist attractions or restaurants',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            location: { type: 'string' },
                            category: { type: 'string', enum: ['tourist_attractions', 'restaurants', 'hotels', 'entertainment', 'nature', 'shopping', 'religion'] },
                            radius: { type: 'number', default: 5000 }
                        },
                        required: ['location']
                    }
                }
            ]
        };
    });
    // --- TOOL EXECUTION ---
    server.setRequestHandler(CallToolRequestSchema, async (request) => {
        const name = request.params.name;
        const args = request.params.arguments;
        console.error(`🛠️ EXECUTING TOOL: ${name}`);
        try {
            if (name === 'search_flights')
                return { content: [{ type: 'text', text: JSON.stringify(await flightService.searchFlights(args)) }] };
            if (name === 'get_weather_forecast')
                return { content: [{ type: 'text', text: JSON.stringify(await weatherService.getWeatherForecast(args)) }] };
            if (name === 'calculate_trip_budget')
                return { content: [{ type: 'text', text: JSON.stringify(await calculateBudget(args)) }] };
            if (name === 'search_places')
                return { content: [{ type: 'text', text: JSON.stringify(await placesService.searchPlaces(args.location, args.category, args.radius)) }] };
            return { content: [{ type: 'text', text: "Tool executed" }] };
        }
        catch (error) {
            return { content: [{ type: 'text', text: `Error: ${error}` }] };
        }
    });
    // Connect transport (this will write the endpoint event)
    try {
        await server.connect(transport);
        console.error('✅ Server connected to transport');
    }
    catch (err) {
        console.error('❌ Connection error during server.connect():', err);
        if (!res.writableEnded)
            res.end();
        return;
    }
    // Get sessionId from transport
    const sessionId = transport.sessionId;
    console.error(`📍 Session ID from transport: ${sessionId}`);
    if (!sessionId) {
        console.error('❌ ERROR: No sessionId assigned by SDK');
        console.error('⚠️ Transport object keys:', Object.keys(transport));
        if (!res.writableEnded)
            res.end();
        return;
    }
    sessions.set(sessionId, transport);
    console.error(`✅ Session registered: ${sessionId}`);
    // Aggressive heartbeat - keep connection ALIVE
    let heartbeatCount = 0;
    const keepAlive = setInterval(() => {
        if (res.writable && !res.writableEnded) {
            res.write(':\n\n');
            if (typeof res.flush === 'function') {
                res.flush();
            }
            heartbeatCount++;
        }
        else {
            console.error(`⚠️ Response no longer writable (heartbeat #${heartbeatCount})`);
            clearInterval(keepAlive);
        }
    }, 3000);
    // Handle all possible connection end scenarios
    req.on('close', () => {
        console.error('⚠️ Request closed by client');
        sessions.delete(sessionId);
        clearInterval(keepAlive);
        try {
            server.close();
        }
        catch (e) {
            console.error('Error closing server:', e);
        }
    });
    req.on('error', (err) => {
        console.error('⚠️ Request error:', err.message);
        sessions.delete(sessionId);
        clearInterval(keepAlive);
        try {
            server.close();
        }
        catch (e) {
            console.error('Error closing server:', e);
        }
    });
    res.on('error', (err) => {
        console.error('⚠️ Response error:', err.message);
        sessions.delete(sessionId);
        clearInterval(keepAlive);
        try {
            server.close();
        }
        catch (e) {
            console.error('Error closing server:', e);
        }
    });
    res.on('finish', () => {
        console.error('⚠️ Response finished unexpectedly');
        sessions.delete(sessionId);
        clearInterval(keepAlive);
        try {
            server.close();
        }
        catch (e) {
            console.error('Error closing server:', e);
        }
    });
});
// --- MESSAGE HANDLER ---
const handleMessage = async (req, res) => {
    const sessionId = req.query.sessionId;
    if (!sessionId) {
        console.error(`❌ No sessionId in query. Query:`, req.query);
        res.status(400).json({ error: 'Missing sessionId parameter' });
        return;
    }
    if (!sessions.has(sessionId)) {
        console.error(`❌ Unknown session: ${sessionId}`);
        console.error(`   Available sessions: ${Array.from(sessions.keys()).join(', ')}`);
        res.status(404).json({ error: 'Session not found', received: sessionId });
        return;
    }
    const transport = sessions.get(sessionId);
    try {
        await transport.handlePostMessage(req, res);
        console.error(`✅ Message handled for session: ${sessionId}`);
    }
    catch (err) {
        console.error(`⚠️ Message handling error for ${sessionId}:`, err);
        res.status(500).json({ error: 'Message handling failed' });
    }
};
app.post('/message', handleMessage);
app.post('/sse', handleMessage);
async function calculateBudget(params) {
    return { totalBudget: 5000, currency: 'USD' };
}
const PORT = 3000;
app.listen(PORT, () => {
    console.error(`✅ Travel MCP Server listening on port ${PORT}`);
});
// Prevent silent crashes
process.on('uncaughtException', (err) => {
    console.error('💥 UNCAUGHT EXCEPTION:', err);
    process.exit(1);
});
process.on('unhandledRejection', (reason, promise) => {
    console.error('💥 UNHANDLED REJECTION:', reason);
});
//# sourceMappingURL=index.js.map