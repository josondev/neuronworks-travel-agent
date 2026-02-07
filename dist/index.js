#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

import { FlightService } from './services/FlightService.js';
import { AccommodationService } from './services/AccommodationService.js';
import { CurrencyService } from './services/CurrencyService.js';
import { WeatherService } from './services/WeatherService.js';
import { PlacesService } from './services/PlacesService.js';

const app = express();
app.use(cors({ origin: '*', methods: ['GET', 'POST', 'OPTIONS'] }));

// Services
const flightService = new FlightService();
const accommodationService = new AccommodationService();
const currencyService = new CurrencyService();
const weatherService = new WeatherService();
const placesService = new PlacesService();

const sessions = new Map();

app.get('/', (req, res) => {
  res.status(200).send('Travel MCP Server is Running 🚀');
});

// --- SSE ENDPOINT ---
app.get('/sse', async (req, res) => {
  console.log('🔗 New SSE Connection');

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
    'Access-Control-Allow-Origin': '*' 
  });

  // 1. Create Transport
  const transport = new SSEServerTransport('/message', res);

  // 2. FIX: SAVE SESSION IMMEDIATELY (Before waiting for anything else)
  // This prevents the "Session not found" race condition
  if (transport.sessionId) {
      sessions.set(transport.sessionId, transport);
      console.log(`✅ Session Created: ${transport.sessionId}`);
  }

  const server = new Server(
    { name: 'travel-planner-server', version: '0.1.0' },
    { capabilities: { tools: {} } }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => {
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
          name: 'search_hotels',
          description: 'Search for hotels',
          inputSchema: {
            type: 'object',
            properties: { city: { type: 'string' } },
            required: ['city']
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
          },
          {
            name: 'get_exchange_rate',
            description: 'Convert currency',
            inputSchema: {
              type: 'object',
              properties: {
                from: { type: 'string' },
                to: { type: 'string' },
                amount: { type: 'number', default: 1 }
              },
              required: ['from', 'to']
            }
          }
      ]
    };
  });

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = request.params.name;
    const args = request.params.arguments;
    console.log(`🛠️ EXECUTING TOOL: ${name}`);
    
    try {
      if (name === 'search_flights') return { content: [{ type: 'text', text: JSON.stringify(await flightService.searchFlights(args)) }] };
      if (name === 'search_hotels') return { content: [{ type: 'text', text: JSON.stringify(await accommodationService.searchAccommodation(args)) }] };
      if (name === 'get_weather_forecast') return { content: [{ type: 'text', text: JSON.stringify(await weatherService.getWeatherForecast(args)) }] };
      if (name === 'calculate_trip_budget') return { content: [{ type: 'text', text: JSON.stringify(await calculateBudget(args)) }] };
      if (name === 'search_places') return { content: [{ type: 'text', text: JSON.stringify(await placesService.searchPlaces(args.location, args.category, args.radius)) }] };
      if (name === 'get_exchange_rate') return { content: [{ type: 'text', text: JSON.stringify(await currencyService.getExchangeRate(args)) }] };
      
      return { content: [{ type: 'text', text: "Tool executed" }] }; 
    } catch (error) {
      return { content: [{ type: 'text', text: `Error: ${error.message}` }] };
    }
  });

  await server.connect(transport);

  req.on('close', () => {
     console.log('⚠️ Connection Closed');
     if (transport.sessionId) sessions.delete(transport.sessionId);
     server.close();
  });
});

// --- MESSAGE HANDLER ---
const handleMessage = async (req, res) => {
  const sessionId = req.query.sessionId;
  
  if (!sessionId || !sessions.has(sessionId)) {
     // console.log(`Ignored message for unknown session: ${sessionId}`);
     res.status(404).send('Session not found');
     return;
  }

  const transport = sessions.get(sessionId);
  try {
      await transport.handlePostMessage(req, res);
  } catch (err) {
      console.error('⚠️ Message handling error:', err);
      res.status(500).send(err.message);
  }
};

app.post('/message', handleMessage);
app.post('/sse', handleMessage);

// Helper for Budget
async function calculateBudget(params) {
    const { budgetLevel = 'mid-range', duration = 7, travelers = 1 } = params;
    const m = {
        budget: { daily: 50, accommodation: 60, flight: 300 },
        'mid-range': { daily: 100, accommodation: 120, flight: 500 },
        'luxury':    { daily: 250, accommodation: 300, flight: 1200 }
    }[budgetLevel] || { daily: 100, accommodation: 120, flight: 500 };
    
    const flights = m.flight * travelers;
    const accommodation = m.accommodation * duration;
    const daily = m.daily * duration * travelers;
    
    return {
        totalBudget: flights + accommodation + daily,
        breakdown: { flights, accommodation, dailyExpenses: daily },
        currency: 'USD',
        budgetLevel, duration, travelers
    };
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Server listening on port ${PORT}`);
});
