import axios from 'axios';
import dotenv from 'dotenv';
dotenv.config();

export class FlightService {
    constructor() {
        this.apiKey = process.env.SERPAPI_API_KEY || process.env.SERPAPI_KEY || '';
        this.baseUrl = 'https://serpapi.com/search.json';

        // Server-owned aliases. The frontend no longer owns airport codes.
        this.airportAliases = {
            chennai: 'MAA', madras: 'MAA',
            madurai: 'IXM',
            coimbatore: 'CJB',
            colombo: 'CMB',
            bangalore: 'BLR', bengaluru: 'BLR',
            hyderabad: 'HYD',
            delhi: 'DEL', 'new delhi': 'DEL',
            mumbai: 'BOM', bombay: 'BOM',
            kochi: 'COK',
            ooty: 'CJB', udhagamandalam: 'CJB',
            kodaikanal: 'IXM',
            goa: 'GOI',
            jaipur: 'JAI',
            ahmedabad: 'AMD',
            pune: 'PNQ',
            kolkata: 'CCU',
            dubai: 'DXB',
            singapore: 'SIN',
            paris: 'CDG',
            london: 'LHR',
            rome: 'FCO',
            tokyo: 'HND',
            'new york': 'JFK'
        };

        if (!this.apiKey) {
            console.warn('⚠️ FlightService: SERPAPI_API_KEY/SERPAPI_KEY is not configured.');
        }
    }

    async resolveAirportCode(value) {
        const raw = String(value || '').trim();
        if (!raw) throw new Error('Airport/city is required for flight search.');

        if (/^[A-Za-z]{3}$/.test(raw)) {
            return raw.toUpperCase();
        }

        const alias = this.airportAliases[raw.toLowerCase()];
        if (alias) return alias;

        if (!this.apiKey) {
            throw new Error(`Could not resolve airport for "${raw}" because SERPAPI_API_KEY is not configured.`);
        }

        // Server-side fallback: use SerpApi Google Search to resolve an unfamiliar
        // city to an IATA airport code. The LLM/client is never involved here.
        const response = await axios.get(this.baseUrl, {
            params: {
                engine: 'google',
                api_key: this.apiKey,
                q: `${raw} nearest commercial airport IATA code`,
                num: 5,
                hl: 'en'
            },
            timeout: 7000
        });

        const chunks = [];
        for (const item of (response.data?.organic_results || []).slice(0, 5)) {
            chunks.push(String(item.title || ''));
            chunks.push(String(item.snippet || ''));
            chunks.push(String(item.link || ''));
        }
        const text = chunks.join(' ');

        const matches = [
            ...text.matchAll(/(?:IATA(?:\s*(?:code|airport code))?\s*[:\-]?\s*|\bIATA\s+)([A-Z]{3})\b/gi),
            ...text.matchAll(/\(([A-Z]{3})\)\s*(?:International|Intl)?\s*Airport/gi)
        ];

        for (const match of matches) {
            const code = String(match[1] || '').toUpperCase();
            if (/^[A-Z]{3}$/.test(code) && !['IATA', 'THE', 'AIR'].includes(code)) {
                console.error(`🧭 Resolved airport "${raw}" → ${code} via SerpApi search`);
                return code;
            }
        }

        throw new Error(`Could not resolve a practical commercial airport for "${raw}".`);
    }

    async searchFlights(params = {}) {
        if (!this.apiKey) {
            return {
                error: 'Flight search unavailable: SERPAPI_API_KEY (or SERPAPI_KEY) is not configured on the server.',
                results: []
            };
        }

        const departDate = String(params.departDate || '').trim();
        const returnDate = params.returnDate ? String(params.returnDate).trim() : '';
        const passengers = Math.max(1, Number(params.passengers) || 1);

        if (!departDate || !/^\d{4}-\d{2}-\d{2}$/.test(departDate)) {
            return { error: 'Flight search requires departDate in YYYY-MM-DD format.', results: [] };
        }

        try {
            const [origin, destination] = await Promise.all([
                this.resolveAirportCode(params.origin),
                this.resolveAirportCode(params.destination)
            ]);

            const requestParams = {
                engine: 'google_flights',
                api_key: this.apiKey,
                departure_id: origin,
                arrival_id: destination,
                outbound_date: departDate,
                type: returnDate ? 1 : 2,
                adults: passengers,
                travel_class: 1,
                sort_by: 2,
                currency: 'USD',
                gl: 'in',
                hl: 'en'
            };

            if (returnDate) requestParams.return_date = returnDate;

            const response = await axios.get(this.baseUrl, {
                params: requestParams,
                timeout: 30000
            });

            if (response.data?.error) throw new Error(response.data.error);

            const flights = this.transformSerpApiResponse(response.data);

            if (!flights.length) {
                return {
                    error: `No live Google Flights results were returned for ${origin} → ${destination} on ${departDate}${returnDate ? ` to ${returnDate}` : ''}.`,
                    results: [],
                    searchLink: response.data?.search_metadata?.google_flights_url || null,
                    source: 'Google Flights via SerpApi'
                };
            }

            return flights;
        } catch (error) {
            const providerError = error.response?.data?.error || error.response?.data?.message || error.message;
            console.error('❌ Google Flights / SerpApi error:', error.response?.data || error.message);
            return {
                error: `Live flight search failed: ${providerError}`,
                results: []
            };
        }
    }

    transformSerpApiResponse(data = {}) {
        const rawFlights = [
            ...(Array.isArray(data.best_flights) ? data.best_flights : []),
            ...(Array.isArray(data.other_flights) ? data.other_flights : [])
        ];

        const uniqueFlights = new Map();
        const googleFlightsUrl = data.search_metadata?.google_flights_url || null;

        for (const offer of rawFlights) {
            const segments = Array.isArray(offer.flights) ? offer.flights : [];
            if (!segments.length) continue;

            const firstSegment = segments[0];
            const lastSegment = segments[segments.length - 1];
            const price = Number(offer.price);
            if (!Number.isFinite(price)) continue;

            const departure = firstSegment.departure_airport?.time || '';
            const arrival = lastSegment.arrival_airport?.time || '';
            const airlines = [...new Set(segments.map(s => s.airline).filter(Boolean))];
            const durationMinutes = segments.reduce((total, segment) => total + (Number(segment.duration) || 0), 0);
            const duration = durationMinutes > 0
                ? `${Math.floor(durationMinutes / 60)}h ${durationMinutes % 60}m`
                : 'Unknown';
            const key = [airlines.join(','), firstSegment.flight_number || '', departure, arrival, price].join('|');

            if (!uniqueFlights.has(key)) {
                uniqueFlights.set(key, {
                    airline: airlines.join(', ') || 'Unknown airline',
                    price,
                    currency: data.search_parameters?.currency || 'USD',
                    departure,
                    arrival,
                    duration,
                    stops: Math.max(0, segments.length - 1),
                    searchLink: googleFlightsUrl,
                    source: 'Google Flights via SerpApi'
                });
            }
        }

        return Array.from(uniqueFlights.values())
            .sort((a, b) => a.price - b.price)
            .slice(0, 10);
    }
}
