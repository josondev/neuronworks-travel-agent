import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

export class FlightService {
    constructor() {
        // SerpApi replaces the retired Amadeus Self-Service integration.
        this.apiKey = process.env.SERPAPI_API_KEY || '';
        this.baseUrl = 'https://serpapi.com/search.json';

        if (!this.apiKey) {
            console.warn('⚠️ SERPAPI_API_KEY is not configured. Flight search will return no results.');
        }
    }

    async searchFlights(params) {
        if (!this.apiKey) {
            console.error('❌ Flight search unavailable: SERPAPI_API_KEY is missing.');
            return [];
        }

        try {
            const requestParams = {
                engine: 'google_flights',
                api_key: this.apiKey,
                departure_id: String(params.origin || '').toUpperCase(),
                arrival_id: String(params.destination || '').toUpperCase(),
                outbound_date: params.departDate,
                type: params.returnDate ? 1 : 2,
                adults: Math.max(1, Number(params.passengers) || 1),
                travel_class: 1,
                sort_by: 2,
                currency: 'USD',
                gl: 'us',
                hl: 'en'
            };

            if (params.returnDate) {
                requestParams.return_date = params.returnDate;
            }

            const response = await axios.get(this.baseUrl, {
                params: requestParams,
                timeout: 30000
            });

            if (response.data?.error) {
                throw new Error(response.data.error);
            }

            return this.transformSerpApiResponse(response.data);
        } catch (error) {
            console.error(
                '❌ Google Flights / SerpApi error:',
                error.response?.data || error.message
            );

            // IMPORTANT: never return fabricated/mock flights.
            // The agent's zero-hallucination policy requires an empty result
            // when live flight data cannot be retrieved.
            return [];
        }
    }

    transformSerpApiResponse(data) {
        const rawFlights = [
            ...(data.best_flights || []),
            ...(data.other_flights || [])
        ];

        if (rawFlights.length === 0) {
            return [];
        }

        const uniqueFlights = new Map();
        const googleFlightsUrl = data.search_metadata?.google_flights_url || null;

        for (const offer of rawFlights) {
            const segments = Array.isArray(offer.flights) ? offer.flights : [];
            if (segments.length === 0) continue;

            const firstSegment = segments[0];
            const lastSegment = segments[segments.length - 1];
            const price = Number(offer.price);

            if (!Number.isFinite(price)) continue;

            const departure = firstSegment.departure_airport?.time || '';
            const arrival = lastSegment.arrival_airport?.time || '';

            const airlines = [
                ...new Set(
                    segments
                        .map(segment => segment.airline)
                        .filter(Boolean)
                )
            ];

            const durationMinutes = Number(
                segments.reduce(
                    (total, segment) => total + (Number(segment.duration) || 0),
                    0
                )
            );

            const durationHours = Math.floor(durationMinutes / 60);
            const durationRemainingMinutes = durationMinutes % 60;
            const duration = durationMinutes > 0
                ? `${durationHours}h ${durationRemainingMinutes}m`
                : 'Unknown';

            const key = [
                airlines.join(','),
                firstSegment.flight_number || '',
                departure,
                arrival,
                price
            ].join('|');

            if (!uniqueFlights.has(key)) {
                uniqueFlights.set(key, {
                    airline: airlines.join(', ') || 'Unknown airline',
                    price,
                    currency: data.search_parameters?.currency || 'USD',
                    departure,
                    arrival,
                    duration,
                    stops: Math.max(0, segments.length - 1),
                    bookingLink: googleFlightsUrl,
                    source: 'Google Flights via SerpApi'
                });
            }
        }

        return Array.from(uniqueFlights.values())
            .sort((a, b) => a.price - b.price)
            .slice(0, 10);
    }
}
