#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import dns from 'dns';
dns.setDefaultResultOrder('ipv4first');

dotenv.config();

import { FlightService } from './services/FlightService.js';
import { AccommodationService } from './services/AccommodationService.js';
import { CurrencyService } from './services/CurrencyService.js';
import { WeatherService } from './services/WeatherService.js';
import { PlacesService } from './services/PlacesService.js';
import { TripPlannerService } from './services/TripPlannerService.js';

const app = express();
app.use(cors({ origin: '*', methods: ['GET', 'POST', 'OPTIONS'] }));

const flightService = new FlightService();
const accommodationService = new AccommodationService();
const currencyService = new CurrencyService();
const weatherService = new WeatherService();
const placesService = new PlacesService();

const sessions = new Map();

app.get('/', (req, res) => {
  res.status(200).send('Travel MCP Server is Running');
});

const textResult = (value) => ({
  content: [{ type: 'text', text: JSON.stringify(value) }]
});

const errorResult = (message) => ({
  isError: true,
  content: [{ type: 'text', text: JSON.stringify({ error: message }) }]
});

function isValidDate(value) {
  return typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value);
}

function requireString(args, name) {
  if (!args || typeof args[name] !== 'string' || !args[name].trim()) {
    throw new Error(`Missing required string argument: ${name}`);
  }
}

function requireNumber(args, name, minimum = 0) {
  const value = Number(args?.[name]);
  if (!Number.isFinite(value) || value < minimum) {
    throw new Error(`Invalid numeric argument: ${name}`);
  }
  return value;
}

app.get('/sse', async (req, res) => {
  console.error('🔗 NEW CONNECTION: Client connected via SSE');

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
    'Access-Control-Allow-Origin': '*'
  });

  res.writeHead = () => res;
  res.write(':' + ' '.repeat(4096) + '\n\n');

  const transport = new SSEServerTransport('/message', res);

  const server = new Server(
    { name: 'travel-planner-server', version: '0.3.0' },
    { capabilities: { tools: {} } }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: 'build_trip_data',
        description: 'Build a complete live trip-data bundle. Pass plain origin/destination city names; the MCP flight service resolves practical airport codes server-side before querying SerpApi. The tool orchestrates flights, hotels, attractions, restaurants, weather and budget, and optionally currency.',
        inputSchema: {
          type: 'object',
          properties: {
            origin: { type: 'string', description: 'Origin city or airport, e.g. Chennai' },
            destinationAirport: { type: 'string', description: 'Optional server-resolved airport code; omit when providing destinationCity.' },
            destinationCity: { type: 'string', description: 'Destination city, e.g. Colombo' },
            destinationCountry: { type: 'string', description: 'Destination country, e.g. Sri Lanka' },
            departDate: { type: 'string', description: 'Departure date in YYYY-MM-DD format' },
            returnDate: { type: 'string', description: 'Return date in YYYY-MM-DD format' },
            passengers: { type: 'number', minimum: 1, default: 1 },
            budgetLevel: { type: 'string', enum: ['budget', 'mid-range', 'luxury'], default: 'budget' },
            currencyFrom: { type: 'string' },
            currencyTo: { type: 'string' },
            currencyAmount: { type: 'number', minimum: 0, default: 1 },
            placesRadius: { type: 'number', minimum: 1, default: 5000 }
          },
          required: ['origin', 'destinationCity', 'destinationCountry', 'departDate', 'returnDate', 'passengers', 'budgetLevel']
        }
      },
      {
        name: 'search_flights',
        description: 'Search live flight prices. The FlightService resolves plain city names or accepts IATA/airport IDs server-side.',
        inputSchema: {
          type: 'object',
          properties: {
            origin: { type: 'string' },
            destination: { type: 'string' },
            departDate: { type: 'string' },
            returnDate: { type: 'string' },
            passengers: { type: 'number', minimum: 1, default: 1 }
          },
          required: ['origin', 'destination', 'departDate']
        }
      },
      {
        name: 'get_weather_forecast',
        description: 'Get live weather forecast data.',
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
        description: 'Return a generic budget estimate, not a live booking total.',
        inputSchema: {
          type: 'object',
          properties: {
            destinations: { type: 'array', items: { type: 'string' }, minItems: 1 },
            duration: { type: 'number', minimum: 1 },
            travelers: { type: 'number', minimum: 1, default: 1 },
            budgetLevel: { type: 'string', enum: ['budget', 'mid-range', 'luxury'] }
          },
          required: ['destinations', 'duration', 'budgetLevel']
        }
      },
      {
        name: 'search_places',
        description: 'Search real places near a location.',
        inputSchema: {
          type: 'object',
          properties: {
            location: { type: 'string' },
            category: { type: 'string', enum: ['tourist_attractions', 'restaurants', 'hotels', 'entertainment', 'nature', 'shopping', 'religion'] },
            radius: { type: 'number', minimum: 1, default: 5000 }
          },
          required: ['location']
        }
      },
      {
        name: 'search_hotels',
        description: 'Search live hotel availability/pricing for a city.',
        inputSchema: {
          type: 'object',
          properties: {
            city: { type: 'string' },
            checkIn: { type: 'string' },
            checkOut: { type: 'string' },
            adults: { type: 'number', minimum: 1, default: 1 }
          },
          required: ['city', 'checkIn', 'checkOut']
        }
      },
      {
        name: 'get_exchange_rate',
        description: 'Get a live exchange rate and convert an amount.',
        inputSchema: {
          type: 'object',
          properties: {
            from: { type: 'string', minLength: 3 },
            to: { type: 'string', minLength: 3 },
            amount: { type: 'number', minimum: 0, default: 1 }
          },
          required: ['from', 'to']
        }
      }
    ]
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = request.params.name;
    const args = request.params.arguments || {};
    console.error(`🛠️ EXECUTING TOOL: ${name}`);
    console.error('📦 TOOL ARGS:', JSON.stringify(args));

    try {
      if (name === 'build_trip_data') {
        for (const field of ['origin', 'destinationCity', 'destinationCountry', 'departDate', 'returnDate']) {
          requireString(args, field);
        }
        if (!isValidDate(args.departDate) || !isValidDate(args.returnDate)) {
          throw new Error('Trip dates must be YYYY-MM-DD');
        }
        if (new Date(`${args.returnDate}T00:00:00Z`) <= new Date(`${args.departDate}T00:00:00Z`)) {
          throw new Error('returnDate must be after departDate');
        }
        const passengers = requireNumber(args, 'passengers', 1);
        if (!['budget', 'mid-range', 'luxury'].includes(args.budgetLevel)) {
          throw new Error('budgetLevel must be budget, mid-range, or luxury');
        }
        return textResult(await tripPlannerService.buildTripData({ ...args, passengers }));
      }

      if (name === 'search_flights') {
        requireString(args, 'origin');
        requireString(args, 'destination');
        requireString(args, 'departDate');
        if (!isValidDate(args.departDate)) throw new Error('departDate must be YYYY-MM-DD');
        if (args.returnDate && !isValidDate(args.returnDate)) throw new Error('returnDate must be YYYY-MM-DD');
        requireNumber(args, 'passengers', 1);
        return textResult(await flightService.searchFlights(args));
      }

      if (name === 'get_weather_forecast') {
        requireString(args, 'city');
        requireString(args, 'country');
        requireString(args, 'startDate');
        requireString(args, 'endDate');
        if (!isValidDate(args.startDate) || !isValidDate(args.endDate)) throw new Error('Weather dates must be YYYY-MM-DD');
        return textResult(await weatherService.getWeatherForecast(args));
      }

      if (name === 'calculate_trip_budget') {
        if (!Array.isArray(args.destinations) || args.destinations.length === 0) throw new Error('destinations must be a non-empty array of strings');
        const duration = requireNumber(args, 'duration', 1);
        const travelers = Number(args.travelers ?? 1);
        if (!Number.isFinite(travelers) || travelers < 1) throw new Error('travelers must be >= 1');
        if (!['budget', 'mid-range', 'luxury'].includes(args.budgetLevel)) throw new Error('budgetLevel must be budget, mid-range, or luxury');
        return textResult(await calculateBudget({ ...args, duration, travelers }));
      }

      if (name === 'search_places') {
        requireString(args, 'location');
        const radius = Number(args.radius ?? 5000);
        if (!Number.isFinite(radius) || radius <= 0) throw new Error('radius must be > 0');
        return textResult(await placesService.searchPlaces(args.location, args.category, radius));
      }

      if (name === 'search_hotels') {
        requireString(args, 'city');
        requireString(args, 'checkIn');
        requireString(args, 'checkOut');
        if (!isValidDate(args.checkIn) || !isValidDate(args.checkOut)) throw new Error('Hotel dates must be YYYY-MM-DD');
        const adults = Number(args.adults ?? 1);
        if (!Number.isFinite(adults) || adults < 1) throw new Error('adults must be >= 1');
        return textResult(await accommodationService.searchAccommodation({ ...args, adults }));
      }

      if (name === 'get_exchange_rate') {
        requireString(args, 'from');
        requireString(args, 'to');
        const amount = Number(args.amount ?? 1);
        if (!Number.isFinite(amount) || amount < 0) throw new Error('amount must be >= 0');
        return textResult(await currencyService.getExchangeRate({ ...args, from: args.from.toUpperCase(), to: args.to.toUpperCase(), amount }));
      }

      return errorResult(`Unknown tool: ${name}`);
    } catch (error) {
      console.error(`❌ Tool ${name} failed:`, error?.message || error);
      return errorResult(error?.message || String(error));
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

const handleMessage = async (req, res) => {
  const sessionId = req.query.sessionId;
  if (!sessionId || !sessions.has(sessionId)) {
    console.error(`❌ Msg received for unknown session: ${sessionId}`);
    res.status(404).send('Session not found');
    return;
  }
  try {
    await sessions.get(sessionId).handlePostMessage(req, res);
    console.error('✅ Message handled');
  } catch (err) {
    console.error('⚠️ Message handling error:', err);
    if (!res.headersSent) res.status(500).send('Message handling failed');
  }
};

app.post('/message', handleMessage);
app.post('/sse', handleMessage);

async function calculateBudget(params) {
  console.error('💰 Calculating generic budget estimate:', params);
  const duration = Number(params.duration);
  const travelers = Number(params.travelers ?? 1);
  const budgetLevel = params.budgetLevel;
  const rates = {
    budget: { daily: 50, hotel: 80, flight: 400 },
    'mid-range': { daily: 150, hotel: 180, flight: 900 },
    luxury: { daily: 500, hotel: 500, flight: 2500 }
  };
  const rate = rates[budgetLevel];
  const flightTotal = rate.flight * travelers;
  const hotelTotal = rate.hotel * duration;
  const dailyTotal = rate.daily * duration * travelers;
  const total = flightTotal + hotelTotal + dailyTotal;
  return {
    type: 'generic_estimate', currency: 'USD', total_budget: total,
    breakdown: { flights_estimate: flightTotal, accommodation_estimate: hotelTotal, daily_expenses_estimate: dailyTotal },
    assumptions: ['Generic estimate only; not a live quote.', 'Does not use actual flight or hotel prices.', 'Actual trip cost should be calculated separately from live tool results.'],
    summary: `Generic ${budgetLevel} estimate for ${travelers} traveler(s) for ${duration} day(s).`
  };
}

const tripPlannerService = new TripPlannerService({
  flightService,
  accommodationService,
  placesService,
  weatherService,
  currencyService,
  calculateBudget
});

const PORT = Number(process.env.PORT || 3000);
app.listen(PORT, () => console.error(`✅ Travel MCP Server listening on port ${PORT}`));
process.on('uncaughtException', (err) => console.error('💥 UNCAUGHT EXCEPTION:', err));
process.on('unhandledRejection', (err) => console.error('💥 UNHANDLED REJECTION:', err));
