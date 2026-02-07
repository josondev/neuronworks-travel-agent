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

// Allow CORS
app.use(cors({ origin: '*', methods: ['GET', 'POST', 'OPTIONS'] }));

// Instantiate Services
const flightService = new FlightService();
const accommodationService = new AccommodationService();
const currencyService = new CurrencyService();
const weatherService = new WeatherService();
const placesService = new PlacesService();
const transports = {};

const sessions = new Map();

app.get('/', (req, res) => {
  res.status(200).send('Travel MCP Server is Running');
});

// --- SSE ENDPOINT ---
app.get('/sse', async (req, res) => {
  console.error('🔗 NEW CONNECTION: Client connected via SSE');

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
    'Access-Control-Allow-Origin': '*' 
  });

  res.writeHead = () => { return res; };
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
        // 1. FLIGHTS
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
        // 2. WEATHER
        {
            name: 'get_weather_forecast',
            description: 'Get weather forecast',
            inputSchema: {
                type: 'object',
                properties: { city: { type: 'string' }, country: { type: 'string' }, startDate: { type: 'string' }, endDate: { type: 'string' } },
                required: ['city', 'country', 'startDate', 'endDate']
            }
        },
        // 3. BUDGET (Now using real logic)
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
          // 4. PLACES (Attractions/Restaurants)
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
          // 5. ACCOMMODATION (This was missing!)
          {
            name: 'search_hotels',
            description: 'Search for hotels in a city',
            inputSchema: {
              type: 'object',
              properties: {
                city: { type: 'string' }
              },
              required: ['city']
            }
          },
          // 6. CURRENCY (This was missing!)
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
    console.error(`🛠️ EXECUTING TOOL: ${name}`);
    
    try {
      if (name === 'search_flights') return { content: [{ type: 'text', text: JSON.stringify(await flightService.searchFlights(args)) }] };
      if (name === 'get_weather_forecast') return { content: [{ type: 'text', text: JSON.stringify(await weatherService.getWeatherForecast(args)) }] };
      if (name === 'calculate_trip_budget') return { content: [{ type: 'text', text: JSON.stringify(await calculateBudget(args)) }] };
      if (name === 'search_places') return { content: [{ type: 'text', text: JSON.stringify(await placesService.searchPlaces(args.location, args.category, args.radius)) }] };
      
      // NEW HANDLERS ADDED HERE
      if (name === 'search_hotels') return { content: [{ type: 'text', text: JSON.stringify(await accommodationService.searchAccommodation(args)) }] };
      if (name === 'get_exchange_rate') return { content: [{ type: 'text', text: JSON.stringify(await currencyService.getExchangeRate(args)) }] };

      return { content: [{ type: 'text', text: "Tool executed" }] }; 
    } catch (error) {
      return { content: [{ type: 'text', text: `Error: ${error}` }] };
    }
  });

  await server.connect(transport);

  const sessionId = transport.sessionId;
  if (sessionId) {
      sessions.set(sessionId, transport);
      console.error(`✅ Session Started: ${sessionId}`);
  }

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


  
  if (!sessionId || !sessions.has(sessionId)) {
     console.error(`❌ Msg received for unknown session: ${sessionId}`);
     res.status(404).send('Session not found');
     return;
  }

  const transport = sessions.get(sessionId);
  try {

  }
};



// --- REAL BUDGET CALCULATOR ---
async function calculateBudget(params) {
    console.error("💰 Calculating budget for:", params);

    const duration = params.duration || 5;
    const travelers = params.travelers || 1;
    const budgetLevel = params.budgetLevel || 'mid-range';
    
    // Estimated costs per level (USD)
    const rates = {
        'budget':    { daily: 50,  hotel: 80,  flight: 400 },
        'mid-range': { daily: 150, hotel: 180, flight: 900 },
        'luxury':    { daily: 500, hotel: 500, flight: 2500 }
    };
    
    const rate = rates[budgetLevel] || rates['mid-range'];
    
    const flightTotal = rate.flight * travelers;
    const hotelTotal = rate.hotel * duration; 
    const dailyTotal = rate.daily * duration * travelers;
    const total = flightTotal + hotelTotal + dailyTotal;

    return {
        currency: 'USD',
        total_budget: total,
        breakdown: {
            flights_estimate: flightTotal,
            accommodation_estimate: hotelTotal,
            daily_expenses_estimate: dailyTotal,
            summary: `Estimated ${budgetLevel} trip for ${travelers} people for ${duration} days.`
        }
    };
}

const PORT = 3000;
app.listen(PORT, () => {

});
