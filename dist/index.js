#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

// 1. SILENCE LOGS (To prevent JSON corruption)
const originalLog = console.log;
console.log = console.error; 

dotenv.config();

// Make sure these paths match where your service files are located
import { FlightService } from './services/FlightService.js';
import { AccommodationService } from './services/AccommodationService.js';
import { CurrencyService } from './services/CurrencyService.js';
import { WeatherService } from './services/WeatherService.js';
import { PlacesService } from './services/PlacesService.js';

const app = express();

// Allow CORS for everyone
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
  res.status(200).send('Travel MCP Server is Running (JS Version + CORS Fixed)');
});

// --- SSE ENDPOINT ---
app.get('/sse', async (req, res) => {
  console.error('🔗 NEW CONNECTION: Client connected via SSE');

  // FIX 1: Manually set headers (including CORS)
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',       // Fixes Cloudflare buffering
    'Access-Control-Allow-Origin': '*' // Fixes "Connect" button in Inspector
  });

  // FIX 2: Monkey Patch (Disable writeHead so SDK doesn't crash Node)
  res.writeHead = () => { return res; };

  // FIX 3: Buffer Buster (Send data immediately)
  res.write(':' + ' '.repeat(4096) + '\n\n');

  const transport = new SSEServerTransport('/message', res);
  
  const server = new Server(
    { name: 'travel-planner-server', version: '0.1.0' },
    { capabilities: { tools: {} } }
  );

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

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = request.params.name;
    const args = request.params.arguments;
    console.error(`🛠️ EXECUTING TOOL: ${name}`);
    
    try {
      if (name === 'search_flights') return { content: [{ type: 'text', text: JSON.stringify(await flightService.searchFlights(args)) }] };
      if (name === 'get_weather_forecast') return { content: [{ type: 'text', text: JSON.stringify(await weatherService.getWeatherForecast(args)) }] };
      if (name === 'calculate_trip_budget') return { content: [{ type: 'text', text: JSON.stringify(await calculateBudget(args)) }] };
      if (name === 'search_places') return { content: [{ type: 'text', text: JSON.stringify(await placesService.searchPlaces(args.location, args.category, args.radius)) }] };
      
      return { content: [{ type: 'text', text: "Tool executed" }] }; 
    } catch (error) {
      return { content: [{ type: 'text', text: `Error: ${error}` }] };
    }
  });

  await server.connect(transport);

  // Capture Session ID
  const sessionId = transport.sessionId;
  if (sessionId) {
      sessions.set(sessionId, transport);
      console.error(`✅ Session Started: ${sessionId}`);
  }

  // Heartbeat Loop
  const keepAlive = setInterval(() => {
    if (res.writable) res.write(':\n\n');
  }, 10000);

  req.on('close', () => {
     console.error('⚠️ Connection Closed');
     if (sessionId) sessions.delete(sessionId);
     clearInterval(keepAlive);
     server.close();
  });
});

// --- MESSAGE HANDLER ---
const handleMessage = async (req, res) => {
  const sessionId = req.query.sessionId;
  
  if (!sessionId || !sessions.has(sessionId)) {
     console.error(`❌ Msg received for unknown session: ${sessionId}`);
     res.status(404).send('Session not found');
     return;
  }

  const transport = sessions.get(sessionId);
  try {
      await transport.handlePostMessage(req, res);
      console.error('✅ Message handled');
  } catch (err) {
      console.error('⚠️ Message handling error:', err);
  }
};

app.post('/message', handleMessage);
app.post('/sse', handleMessage);

async function calculateBudget(params) {
    return { totalBudget: 5000, currency: 'USD' };
}

const PORT = 3000;
app.listen(PORT, () => {
  console.error(`✅ Travel MCP Server (JS Mode) listening on port ${PORT}`);
});

process.on('uncaughtException', (err) => {
  console.error('💥 UNCAUGHT EXCEPTION:', err);
});