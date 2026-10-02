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

// ─── Service imports ─────────────────────────────────────────────────────────
import { FlightService }       from './services/FlightService.js';
import { AccommodationService } from './services/AccommodationService.js';
import { CurrencyService }     from './services/CurrencyService.js';
import { WeatherService }      from './services/WeatherService.js';
import { PlacesService }       from './services/PlacesService.js';
import { TripPlannerService }  from './services/TripPlannerService.js';

// ─── DB layer imports ────────────────────────────────────────────────────────
import { connectDB, disconnectDB, isDBReady } from '../src/db/prisma.js';
import {
  upsertUser,
  logSearch,
  getSearchHistory,
  saveItinerary,
  getSavedItineraries,
  saveBudgetEstimate,
} from '../src/db/DatabaseService.js';
import { withCache, pruneExpiredCache } from '../src/db/CacheService.js';

// ─── Express setup ───────────────────────────────────────────────────────────
const app = express();
app.use(cors({ origin: '*', methods: ['GET', 'POST', 'OPTIONS'] }));
app.use(express.json()); // required so req.body exists for handlePostMessage(req, res, req.body)

// ─── Service instances ────────────────────────────────────────────────────────
const flightService       = new FlightService();
const accommodationService = new AccommodationService();
const currencyService     = new CurrencyService();
const weatherService      = new WeatherService();
const placesService       = new PlacesService();

// SSE session registry: sessionId → { transport, userId }
const sessions = new Map();

// ─── Routes ───────────────────────────────────────────────────────────────────
app.get('/', (req, res) => res.status(200).send('Travel MCP Server is Running'));

app.get('/health', (req, res) => {
  res.status(200).json({
    status: 'ok',
    uptime: process.uptime(),
    db:     isDBReady() ? 'connected' : 'unavailable',
  });
});

// ─── Helpers ──────────────────────────────────────────────────────────────────
const textResult  = (value)   => ({ content: [{ type: 'text', text: JSON.stringify(value) }] });
const errorResult = (message) => ({ isError: true, content: [{ type: 'text', text: JSON.stringify({ error: message }) }] });

function isValidDate(v) { return typeof v === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(v); }

function requireString(args, name) {
  if (!args || typeof args[name] !== 'string' || !args[name].trim())
    throw new Error(`Missing required string argument: ${name}`);
}

function requireNumber(args, name, minimum = 0) {
  const v = Number(args?.[name]);
  if (!Number.isFinite(v) || v < minimum) throw new Error(`Invalid numeric argument: ${name}`);
  return v;
}

// ─── SSE endpoint ─────────────────────────────────────────────────────────────
app.get('/sse', async (req, res) => {
  console.error('🔗 NEW CONNECTION: Client connected via SSE');

  // Let SSEServerTransport.start() own the response headers entirely.
  // Only inject custom headers via setHeader() before handing off.
  res.setHeader('X-Accel-Buffering', 'no');
  res.setHeader('Access-Control-Allow-Origin', '*');

  const transport = new SSEServerTransport('/message', res);

  const server = new Server(
    { name: 'travel-planner-server', version: '0.4.0' },
    { capabilities: { tools: {} } }
  );

  // ── Tool list ──────────────────────────────────────────────────────────────
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      // ── 1. Build full trip bundle ─────────────────────────────────────────
      {
        name: 'build_trip_data',
        description: 'Build a complete live trip-data bundle. Orchestrates flights, hotels, attractions, restaurants, weather, budget, and currency in one call. Results are cached.',
        inputSchema: {
          type: 'object',
          properties: {
            origin:             { type: 'string', description: 'Origin city or airport, e.g. Chennai' },
            destinationAirport: { type: 'string', description: 'Optional IATA code; omit when providing destinationCity.' },
            destinationCity:    { type: 'string', description: 'Destination city, e.g. Colombo' },
            destinationCountry: { type: 'string', description: 'Destination country, e.g. Sri Lanka' },
            departDate:         { type: 'string', description: 'Departure date YYYY-MM-DD' },
            returnDate:         { type: 'string', description: 'Return date YYYY-MM-DD' },
            passengers:         { type: 'number', minimum: 1, default: 1 },
            budgetLevel:        { type: 'string', enum: ['budget', 'mid-range', 'luxury'], default: 'budget' },
            currencyFrom:       { type: 'string' },
            currencyTo:         { type: 'string' },
            currencyAmount:     { type: 'number', minimum: 0, default: 1 },
            placesRadius:       { type: 'number', minimum: 1, default: 5000 },
          },
          required: ['origin', 'destinationCity', 'destinationCountry', 'departDate', 'returnDate', 'passengers', 'budgetLevel'],
        },
      },
      // ── 2. Flights ────────────────────────────────────────────────────────
      {
        name: 'search_flights',
        description: 'Search live flight prices via SerpApi Google Flights. City names are resolved to IATA codes server-side. Results are cached 30 min.',
        inputSchema: {
          type: 'object',
          properties: {
            origin:      { type: 'string' },
            destination: { type: 'string' },
            departDate:  { type: 'string' },
            returnDate:  { type: 'string' },
            passengers:  { type: 'number', minimum: 1, default: 1 },
          },
          required: ['origin', 'destination', 'departDate'],
        },
      },
      // ── 3. Weather ────────────────────────────────────────────────────────
      {
        name: 'get_weather_forecast',
        description: 'Get live weather forecast via OpenWeatherMap. Results are cached 2 hours.',
        inputSchema: {
          type: 'object',
          properties: {
            city:      { type: 'string' },
            country:   { type: 'string' },
            startDate: { type: 'string' },
            endDate:   { type: 'string' },
          },
          required: ['city', 'country', 'startDate', 'endDate'],
        },
      },
      // ── 4. Budget ─────────────────────────────────────────────────────────
      {
        name: 'calculate_trip_budget',
        description: 'Return a generic budget estimate (not a live booking total). Results are stored in DB per session.',
        inputSchema: {
          type: 'object',
          properties: {
            destinations: { type: 'array', items: { type: 'string' }, minItems: 1 },
            duration:     { type: 'number', minimum: 1 },
            travelers:    { type: 'number', minimum: 1, default: 1 },
            budgetLevel:  { type: 'string', enum: ['budget', 'mid-range', 'luxury'] },
          },
          required: ['destinations', 'duration', 'budgetLevel'],
        },
      },
      // ── 5. Places ─────────────────────────────────────────────────────────
      {
        name: 'search_places',
        description: 'Search real places near a location via Geoapify. Results are cached 24 hours.',
        inputSchema: {
          type: 'object',
          properties: {
            location: { type: 'string' },
            category: { type: 'string', enum: ['tourist_attractions', 'restaurants', 'hotels', 'entertainment', 'nature', 'shopping', 'religion'] },
            radius:   { type: 'number', minimum: 1, default: 5000 },
          },
          required: ['location'],
        },
      },
      // ── 6. Hotels ─────────────────────────────────────────────────────────
      {
        name: 'search_hotels',
        description: 'Search live hotel availability via SerpApi Google Hotels. Results are cached 1 hour.',
        inputSchema: {
          type: 'object',
          properties: {
            city:     { type: 'string' },
            checkIn:  { type: 'string' },
            checkOut: { type: 'string' },
            adults:   { type: 'number', minimum: 1, default: 1 },
          },
          required: ['city', 'checkIn', 'checkOut'],
        },
      },
      // ── 7. Currency ───────────────────────────────────────────────────────
      {
        name: 'get_exchange_rate',
        description: 'Get a live exchange rate and convert an amount via ExchangeRate-API. Results are cached 15 min.',
        inputSchema: {
          type: 'object',
          properties: {
            from:   { type: 'string', minLength: 3 },
            to:     { type: 'string', minLength: 3 },
            amount: { type: 'number', minimum: 0, default: 1 },
          },
          required: ['from', 'to'],
        },
      },
      // ── 8. Search history (DB) ────────────────────────────────────────────
      {
        name: 'get_search_history',
        description: 'Return the last N tool calls made in this session (stored in DB). Returns an empty list when DB is unavailable.',
        inputSchema: {
          type: 'object',
          properties: {
            limit: { type: 'number', minimum: 1, maximum: 50, default: 10 },
          },
          required: [],
        },
      },
      // ── 9. Save itinerary (DB) ────────────────────────────────────────────
      {
        name: 'save_itinerary',
        description: 'Save a named trip plan to the DB for the current session.',
        inputSchema: {
          type: 'object',
          properties: {
            title:       { type: 'string' },
            destination: { type: 'string' },
            startDate:   { type: 'string' },
            endDate:     { type: 'string' },
            notes:       { type: 'string' },
            itinerary:   { type: 'object', description: 'The trip plan data as a JSON object.' },
          },
          required: ['title', 'destination', 'itinerary'],
        },
      },
      // ── 10. Get saved itineraries (DB) ────────────────────────────────────
      {
        name: 'get_saved_itineraries',
        description: 'List all saved trip plans for the current session from the DB.',
        inputSchema: {
          type: 'object',
          properties: {},
          required: [],
        },
      },
    ],
  }));

  // ── Tool handlers ──────────────────────────────────────────────────────────
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const name = request.params.name;
    const args = request.params.arguments || {};
    console.error(`🛠️  EXECUTING TOOL: ${name}`);
    console.error('📦 TOOL ARGS:', JSON.stringify(args));

    // Retrieve userId for this SSE session (null when DB unavailable)
    const sessionEntry = sessions.get(transport.sessionId);
    const userId       = sessionEntry?.userId ?? null;

    try {

      // ── build_trip_data ──────────────────────────────────────────────────
      if (name === 'build_trip_data') {
        for (const f of ['origin', 'destinationCity', 'destinationCountry', 'departDate', 'returnDate'])
          requireString(args, f);
        if (!isValidDate(args.departDate) || !isValidDate(args.returnDate))
          throw new Error('Trip dates must be YYYY-MM-DD');
        if (new Date(`${args.returnDate}T00:00:00Z`) <= new Date(`${args.departDate}T00:00:00Z`))
          throw new Error('returnDate must be after departDate');
        const passengers = requireNumber(args, 'passengers', 1);
        if (!['budget', 'mid-range', 'luxury'].includes(args.budgetLevel))
          throw new Error('budgetLevel must be budget, mid-range, or luxury');

        const result = await withCache(name, args, () =>
          tripPlannerService.buildTripData({ ...args, passengers })
        );
        logSearch(userId, name, args, result).catch(() => {});
        return textResult(result);
      }

      // ── search_flights ───────────────────────────────────────────────────
      if (name === 'search_flights') {
        requireString(args, 'origin');
        requireString(args, 'destination');
        requireString(args, 'departDate');
        if (!isValidDate(args.departDate))
          throw new Error('departDate must be YYYY-MM-DD');
        if (args.returnDate && !isValidDate(args.returnDate))
          throw new Error('returnDate must be YYYY-MM-DD');
        requireNumber(args, 'passengers', 1);

        const result = await withCache(name, args, () => flightService.searchFlights(args));
        logSearch(userId, name, args, result).catch(() => {});
        return textResult(result);
      }

      // ── get_weather_forecast ─────────────────────────────────────────────
      if (name === 'get_weather_forecast') {
        requireString(args, 'city');
        requireString(args, 'country');
        requireString(args, 'startDate');
        requireString(args, 'endDate');
        if (!isValidDate(args.startDate) || !isValidDate(args.endDate))
          throw new Error('Weather dates must be YYYY-MM-DD');

        const result = await withCache(name, args, () => weatherService.getWeatherForecast(args));
        logSearch(userId, name, args, result).catch(() => {});
        return textResult(result);
      }

      // ── calculate_trip_budget ────────────────────────────────────────────
      if (name === 'calculate_trip_budget') {
        if (!Array.isArray(args.destinations) || args.destinations.length === 0)
          throw new Error('destinations must be a non-empty array of strings');
        const duration  = requireNumber(args, 'duration', 1);
        const travelers = Number(args.travelers ?? 1);
        if (!Number.isFinite(travelers) || travelers < 1)
          throw new Error('travelers must be >= 1');
        if (!['budget', 'mid-range', 'luxury'].includes(args.budgetLevel))
          throw new Error('budgetLevel must be budget, mid-range, or luxury');

        const normalized = { ...args, duration, travelers };
        const result     = await withCache(name, normalized, () => calculateBudget(normalized));
        logSearch(userId, name, normalized, result).catch(() => {});

        // Persist budget estimate per session
        if (userId) {
          saveBudgetEstimate(userId, {
            destination:  args.destinations.join(', '),
            durationDays: duration,
            travelers,
            budgetLevel:  args.budgetLevel,
            totalUsd:     result.total_budget,
            breakdown:    result.breakdown,
          }).catch(() => {});
        }

        return textResult(result);
      }

      // ── search_places ────────────────────────────────────────────────────
      if (name === 'search_places') {
        requireString(args, 'location');
        const radius = Number(args.radius ?? 5000);
        if (!Number.isFinite(radius) || radius <= 0) throw new Error('radius must be > 0');

        const result = await withCache(name, args, () =>
          placesService.searchPlaces(args.location, args.category, radius)
        );
        logSearch(userId, name, args, result).catch(() => {});
        return textResult(result);
      }

      // ── search_hotels ────────────────────────────────────────────────────
      if (name === 'search_hotels') {
        requireString(args, 'city');
        requireString(args, 'checkIn');
        requireString(args, 'checkOut');
        if (!isValidDate(args.checkIn) || !isValidDate(args.checkOut))
          throw new Error('Hotel dates must be YYYY-MM-DD');
        const adults = Number(args.adults ?? 1);
        if (!Number.isFinite(adults) || adults < 1) throw new Error('adults must be >= 1');

        const normalized = { ...args, adults };
        const result     = await withCache(name, normalized, () =>
          accommodationService.searchAccommodation(normalized)
        );
        logSearch(userId, name, normalized, result).catch(() => {});
        return textResult(result);
      }

      // ── get_exchange_rate ────────────────────────────────────────────────
      if (name === 'get_exchange_rate') {
        requireString(args, 'from');
        requireString(args, 'to');
        const amount     = Number(args.amount ?? 1);
        if (!Number.isFinite(amount) || amount < 0) throw new Error('amount must be >= 0');
        const normalized = { ...args, from: args.from.toUpperCase(), to: args.to.toUpperCase(), amount };

        const result = await withCache(name, normalized, () =>
          currencyService.getExchangeRate(normalized)
        );
        logSearch(userId, name, normalized, result).catch(() => {});
        return textResult(result);
      }

      // ── get_search_history (DB) ──────────────────────────────────────────
      if (name === 'get_search_history') {
        if (!userId) return textResult({ history: [], note: 'DB unavailable or session not tracked.' });
        const limit   = Math.min(Number(args.limit ?? 10), 50);
        const history = await getSearchHistory(userId, limit);
        return textResult({ history });
      }

      // ── save_itinerary (DB) ──────────────────────────────────────────────
      if (name === 'save_itinerary') {
        requireString(args, 'title');
        requireString(args, 'destination');
        if (!args.itinerary || typeof args.itinerary !== 'object')
          throw new Error('itinerary must be a JSON object');
        if (!userId)
          return textResult({ saved: false, note: 'DB unavailable — itinerary not persisted.' });

        const record = await saveItinerary(userId, {
          title:       args.title,
          destination: args.destination,
          startDate:   args.startDate  || null,
          endDate:     args.endDate    || null,
          notes:       args.notes      || null,
          itinerary:   args.itinerary,
        });
        return textResult({
          saved: !!record,
          id:    record?.id ?? null,
          title: args.title,
        });
      }

      // ── get_saved_itineraries (DB) ───────────────────────────────────────
      if (name === 'get_saved_itineraries') {
        if (!userId) return textResult({ itineraries: [], note: 'DB unavailable or session not tracked.' });
        const itineraries = await getSavedItineraries(userId);
        return textResult({ itineraries });
      }

      return errorResult(`Unknown tool: ${name}`);

    } catch (error) {
      console.error(`❌ Tool ${name} failed:`, error?.message || error);
      return errorResult(error?.message || String(error));
    }
  });

  // Connect transport → start SSE stream + send event:endpoint frame
  await server.connect(transport);

  const sessionId = transport.sessionId;
  if (sessionId) {
    // Upsert a DB user for this session (fire-and-forget, non-blocking)
    upsertUser(sessionId).then(user => {
      sessions.set(sessionId, { transport, userId: user?.id ?? null });
      if (user) console.error(`✅ DB: Session user upserted — userId: ${user.id}`);
    }).catch(() => {
      sessions.set(sessionId, { transport, userId: null });
    });

    // Ensure the session entry exists immediately (userId filled in above async)
    if (!sessions.has(sessionId)) sessions.set(sessionId, { transport, userId: null });
    console.error(`✅ Session Started: ${sessionId}`);
  }

  const keepAlive = setInterval(() => {
    if (res.writable) res.write(':\n\n');
  }, 10000);

  req.on('close', () => {
    console.error('⚠️  Connection Closed');
    if (sessionId) sessions.delete(sessionId);
    clearInterval(keepAlive);
    server.close();
  });
});

// ─── POST /message handler ────────────────────────────────────────────────────
const handleMessage = async (req, res) => {
  const sessionId = req.query.sessionId;
  if (!sessionId || !sessions.has(sessionId)) {
    console.error(`❌ Msg received for unknown session: ${sessionId}`);
    res.status(404).send('Session not found');
    return;
  }
  try {
    // Pass req.body (already parsed by express.json()) as parsedBody so the
    // SDK does not try to re-read the consumed raw stream — fixes 400 errors.
    await sessions.get(sessionId).transport.handlePostMessage(req, res, req.body);
    console.error('✅ Message handled');
  } catch (err) {
    console.error('⚠️  Message handling error:', err);
    if (!res.headersSent) res.status(500).send('Message handling failed');
  }
};

app.post('/message', handleMessage);
app.post('/sse',     handleMessage);

// ─── Budget calculator ────────────────────────────────────────────────────────
async function calculateBudget(params) {
  console.error('💰 Calculating generic budget estimate:', params);
  const duration    = Number(params.duration);
  const travelers   = Number(params.travelers ?? 1);
  const budgetLevel = params.budgetLevel;
  const rates = {
    budget:      { daily: 50,  hotel: 80,  flight: 400  },
    'mid-range': { daily: 150, hotel: 180, flight: 900  },
    luxury:      { daily: 500, hotel: 500, flight: 2500 },
  };
  const rate        = rates[budgetLevel];
  const flightTotal = rate.flight * travelers;
  const hotelTotal  = rate.hotel  * duration;
  const dailyTotal  = rate.daily  * duration * travelers;
  const total       = flightTotal + hotelTotal + dailyTotal;
  return {
    type:         'generic_estimate',
    currency:     'USD',
    total_budget: total,
    breakdown: {
      flights_estimate:       flightTotal,
      accommodation_estimate: hotelTotal,
      daily_expenses_estimate: dailyTotal,
    },
    assumptions: [
      'Generic estimate only; not a live quote.',
      'Does not use actual flight or hotel prices.',
      'Actual trip cost should be calculated separately from live tool results.',
    ],
    summary: `Generic ${budgetLevel} estimate for ${travelers} traveler(s) for ${duration} day(s).`,
  };
}

// ─── TripPlannerService ───────────────────────────────────────────────────────
const tripPlannerService = new TripPlannerService({
  flightService,
  accommodationService,
  placesService,
  weatherService,
  currencyService,
  calculateBudget,
});

// ─── Startup ──────────────────────────────────────────────────────────────────
const PORT = Number(process.env.PORT || 3000);

async function main() {
  // Connect to DB (non-blocking — server starts even without a DB)
  await connectDB();

  // Prune stale cache entries every hour
  setInterval(() => pruneExpiredCache().catch(() => {}), 60 * 60 * 1000);

  app.listen(PORT, () =>
    console.error(`✅ Travel MCP Server listening on port ${PORT}`)
  );
}

main().catch(err => console.error('💥 Startup error:', err));

// ─── Graceful shutdown ────────────────────────────────────────────────────────
async function shutdown(signal) {
  console.error(`\n${signal} received — shutting down gracefully…`);
  await disconnectDB();
  process.exit(0);
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT',  () => shutdown('SIGINT'));
process.on('uncaughtException',   (err) => console.error('💥 UNCAUGHT EXCEPTION:',   err));
process.on('unhandledRejection',  (err) => console.error('💥 UNHANDLED REJECTION:',  err));
