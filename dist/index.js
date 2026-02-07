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

app.use(cors({ 
  origin: '*', 
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type']
}));

app.use(express.json());

const flightService = new FlightService();
const accommodationService = new AccommodationService();
const currencyService = new CurrencyService();
const weatherService = new WeatherService();
const placesService = new PlacesService();
const transports = {};

// Create a single shared MCP server instance
function createMCPServer() {
  const server = new Server(
    { name: 'travel-planner-server', version: '0.1.0' },
    { capabilities: { tools: {} } }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => {
    console.error('📋 Tools list requested');
    return {
      tools: [
        {
          name: 'search_flights',
          description: 'Search for flight prices and schedules',
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
            properties: {
              city: { type: 'string' },
              checkIn: { type: 'string' },
              checkOut: { type: 'string' },
              guests: { type: 'number', default: 2 }
            },
            required: ['city', 'checkIn', 'checkOut']
          }
        },
        {
          name: 'get_weather_forecast',
          description: 'Get weather forecast',
          inputSchema: {
            type: 'object',
            properties: {
              city: { type: 'string' },
              country: { type: 'string' },
              startDate: { type: 'string' },
              endDate: { type: 'string' }
            },
            required: ['city', 'country', 'startDate', 'endDate']
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
          description: 'Search for attractions/restaurants',
          inputSchema: {
            type: 'object',
            properties: {
              location: { type: 'string' },
              category: { type: 'string', enum: ['tourist_attractions', 'restaurants', 'hotels'] },
              radius: { type: 'number', default: 5000 }
            },
            required: ['location']
          }
        }
      ]
    };
  });

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    console.error(`🛠️ Executing: ${name}`);

    try {
      let result;
      
      if (name === 'search_flights') result = await flightService.searchFlights(args);
      else if (name === 'search_hotels') result = await accommodationService.searchAccommodation(args);
      else if (name === 'get_weather_forecast') result = await weatherService.getWeatherForecast(args);
      else if (name === 'get_exchange_rate') result = await currencyService.getExchangeRate(args);
      else if (name === 'calculate_trip_budget') result = await calculateBudget(args);
      else if (name === 'search_places') result = await placesService.searchPlaces(args.location, args.category, args.radius);
      else throw new Error(`Unknown tool: ${name}`);

      return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
    } catch (error) {
      console.error(`❌ Error:`, error);
      return { content: [{ type: 'text', text: `Error: ${error.message}` }] };
    }
  });

  return server;
}

app.get('/', (req, res) => {
  res.json({ status: 'ok', name: 'Travel MCP Server', version: '1.0.0' });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

app.get('/sse', async (req, res) => {
  console.error('🔗 New SSE connection');
  
  const server = createMCPServer();
  const transport = new SSEServerTransport('/messages', res);
  const sessionId = transport.sessionId;
  transports[sessionId] = transport;
  transport.onclose = () => {
    delete transports[sessionId];
  };
  
  try {
    await server.connect(transport);
    console.error('✅ Connected');
  } catch (error) {
    console.error('❌ Connection error:', error);
    if (!res.headersSent) {
      res.status(500).send('Error establishing SSE stream');
    }
  }
  
  req.on('close', () => {
    console.error('⚠️ Disconnected');
    server.close();
  });
});

// Handle POST to /messages for client requests
app.post('/messages', async (req, res) => {
  console.error('📨 POST to /messages');
  const sessionId = req.query.sessionId;
  if (!sessionId || Array.isArray(sessionId)) {
    res.status(400).send('Missing sessionId parameter');
    return;
  }
  const transport = transports[sessionId];
  if (!transport) {
    res.status(404).send('Session not found');
    return;
  }
  try {
    await transport.handlePostMessage(req, res, req.body);
  } catch (error) {
    console.error('❌ Error handling request:', error);
    if (!res.headersSent) {
      res.status(500).send('Error handling request');
    }
  }
});

async function calculateBudget(params) {
  const { budgetLevel = 'mid-range', duration = 7, travelers = 1 } = params;
  const m = {
    budget: { daily: 50, accommodation: 60, flight: 300 },
    'mid-range': { daily: 100, accommodation: 120, flight: 500 },
    luxury: { daily: 250, accommodation: 300, flight: 1200 }
  }[budgetLevel];
  
  const flights = m.flight * travelers * 2;
  const accommodation = m.accommodation * duration;
  const daily = m.daily * duration * travelers;
  
  return {
    totalBudget: flights + accommodation + daily,
    breakdown: { flights, accommodation, dailyExpenses: daily },
    currency: 'USD',
    budgetLevel,
    duration,
    travelers
  };
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.error('🎉 Travel MCP Server running on port', PORT);
});
