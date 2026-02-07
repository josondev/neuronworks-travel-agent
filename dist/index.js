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
  allowedHeaders: ['Content-Type', 'Authorization', 'Accept']
}));

app.use(express.json());

const flightService = new FlightService();
const accommodationService = new AccommodationService();
const currencyService = new CurrencyService();
const weatherService = new WeatherService();
const placesService = new PlacesService();

// Health check
app.get('/', (req, res) => {
  res.json({ 
    status: 'ok', 
    name: 'Travel MCP Server',
    version: '1.0.0',
    endpoints: {
      sse: '/sse',
      health: '/health'
    }
  });
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// SSE ENDPOINT - Let SDK handle everything
app.get('/sse', async (req, res) => {
  console.error('🔗 SSE Connection initiated');
  
  // DON'T set headers manually - let the SDK do it!
  // The SSEServerTransport will handle all header management
  
  // Create a new Server instance for this connection
  const server = new Server(
    { name: 'travel-planner-server', version: '0.1.0' },
    { capabilities: { tools: {} } }
  );

  // Register tool handlers
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    console.error('📋 Tool list requested');
    return {
      tools: [
        {
          name: 'search_flights',
          description: 'Search for flight prices and schedules',
          inputSchema: {
            type: 'object',
            properties: {
              origin: { type: 'string', description: 'Origin airport code' },
              destination: { type: 'string', description: 'Destination airport code' },
              departDate: { type: 'string', description: 'Departure date YYYY-MM-DD' },
              returnDate: { type: 'string', description: 'Return date YYYY-MM-DD' },
              passengers: { type: 'number', default: 1 }
            },
            required: ['origin', 'destination', 'departDate']
          }
        },
        {
          name: 'get_weather_forecast',
          description: 'Get weather forecast for travel dates',
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
          name: 'calculate_trip_budget',
          description: 'Calculate estimated trip budget',
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
              category: { 
                type: 'string', 
                enum: ['tourist_attractions', 'restaurants', 'hotels', 'entertainment', 'nature']
              },
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
    console.error(`🛠️ Tool execution: ${name}`);

    try {
      let result;
      
      switch (name) {
        case 'search_flights':
          result = await flightService.searchFlights(args);
          break;
        case 'get_weather_forecast':
          result = await weatherService.getWeatherForecast(args);
          break;
        case 'calculate_trip_budget':
          result = await calculateBudget(args);
          break;
        case 'search_places':
          result = await placesService.searchPlaces(args.location, args.category, args.radius);
          break;
        default:
          throw new Error(`Unknown tool: ${name}`);
      }

      return {
        content: [{ type: 'text', text: JSON.stringify(result, null, 2) }]
      };
    } catch (error) {
      console.error(`❌ Tool error:`, error);
      return {
        content: [{ type: 'text', text: `Error: ${error.message}` }],
        isError: true
      };
    }
  });

  // Create transport - THIS handles all SSE formatting
  // Use a simple endpoint path
  const transport = new SSEServerTransport('/messages', res);
  
  try {
    // Connect the server to the transport
    // This will write the proper SSE headers and endpoint event
    await server.connect(transport);
    console.error('✅ MCP Server connected');
    
    // DON'T manually write anything to res after this point
    // The SDK handles everything
    
  } catch (error) {
    console.error('❌ Connection error:', error);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to establish SSE connection' });
    }
  }
  
  // Handle connection cleanup
  req.on('close', () => {
    console.error('⚠️ Client disconnected');
    try {
      server.close();
    } catch (e) {
      console.error('Error closing server:', e);
    }
  });
});

// Message handler for POST requests
// The SDK will direct POST messages here based on the endpoint path
app.post('/messages', (req, res) => {
  console.error('📨 Message received:', req.body);
  // The SSEServerTransport handles this automatically
  // Just return 202 Accepted
  res.status(202).json({ status: 'accepted' });
});

async function calculateBudget(params) {
  const { budgetLevel = 'mid-range', duration = 7, travelers = 1 } = params;
  const multipliers = {
    budget: { daily: 50, accommodation: 60, flight: 300 },
    'mid-range': { daily: 100, accommodation: 120, flight: 500 },
    luxury: { daily: 250, accommodation: 300, flight: 1200 }
  };
  
  const m = multipliers[budgetLevel];
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
  console.error('');
  console.error('🎉 ================================');
  console.error('🎉  Travel MCP Server RUNNING!');
  console.error('🎉 ================================');
  console.error('');
  console.error(`📍 Port: ${PORT}`);
  console.error(`🏥 Health: http://localhost:${PORT}/health`);
  console.error(`📡 SSE: http://localhost:${PORT}/sse`);
  console.error('');
  console.error('✅ Ready for connections...');
  console.error('');
});

process.on('uncaughtException', (err) => {
  console.error('💥 Uncaught exception:', err);
});

process.on('unhandledRejection', (reason) => {
  console.error('💥 Unhandled rejection:', reason);
});