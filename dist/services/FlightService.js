import axios from 'axios';
export class FlightService {
    apiKey;
    apiSecret;
    baseUrl = 'https://test.api.amadeus.com/v2'; // Test environment
    // For production, use: https://api.amadeus.com/v2
    accessToken = '';
    tokenExpiry = 0;
    constructor() {
        this.apiKey = process.env.FLIGHT_API_KEY || '';
        this.apiSecret = process.env.FLIGHT_API_SECRET || '';
        if (!this.apiKey || this.apiKey === 'test_key_replace_later') {
            console.warn('Amadeus API key not configured. Flight service will return mock data.');
        }
    }
    /**
     * Get OAuth access token from Amadeus
     */
    async getAccessToken() {
        // Check if we have a valid token
        if (this.accessToken && Date.now() < this.tokenExpiry) {
            return this.accessToken;
        }
        try {
            const response = await axios.post('https://test.api.amadeus.com/v1/security/oauth2/token', new URLSearchParams({
                grant_type: 'client_credentials',
                client_id: this.apiKey,
                client_secret: this.apiSecret,
            }), {
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
            });
            this.accessToken = response.data.access_token;
            // Set expiry to 5 minutes before actual expiry for safety
            this.tokenExpiry = Date.now() + (response.data.expires_in - 300) * 1000;
            return this.accessToken;
        }
        catch (error) {
            console.error('Failed to get Amadeus access token:', error);
            throw new Error('Authentication failed with Amadeus API');
        }
    }
    /**
     * Search for flights using Amadeus Flight Offers Search API
     */
    async searchFlights(params) {
        // Return mock data if API keys are not configured
        if (!this.apiKey || this.apiKey === 'test_key_replace_later') {
            return this.getMockFlightData(params);
        }
        try {
            const token = await this.getAccessToken();
            const searchParams = {
                originLocationCode: params.origin.toUpperCase(),
                destinationLocationCode: params.destination.toUpperCase(),
                departureDate: params.departDate,
                adults: params.passengers,
                max: 10, // Limit results
            };
            // Add return date if provided (for round-trip)
            if (params.returnDate) {
                searchParams.returnDate = params.returnDate;
            }
            // Add travel class if specified
            if (params.travelClass) {
                searchParams.travelClass = params.travelClass;
            }
            const response = await axios.get(`${this.baseUrl}/shopping/flight-offers`, {
                params: searchParams,
                headers: {
                    Authorization: `Bearer ${token}`,
                },
            });
            // Transform Amadeus response to our format
            return this.transformAmadeusResponse(response.data);
        }
        catch (error) {
            console.error('Amadeus API error:', error.response?.data || error.message);
            // Return mock data as fallback
            return this.getMockFlightData(params);
        }
    }
    /**
     * Transform Amadeus API response to our FlightResult format
     */
    /**
     * Transform Amadeus API response to our FlightResult format
     * AND remove duplicates
     */
    transformAmadeusResponse(data) {
        if (!data.data || data.data.length === 0) {
            return [];
        }
        const uniqueFlights = new Map();
        data.data.forEach((offer) => {
            const firstItinerary = offer.itineraries[0];
            const firstSegment = firstItinerary.segments[0];
            const lastSegment = firstItinerary.segments[firstItinerary.segments.length - 1];
            const carrierCode = firstSegment.carrierCode;
            const airline = data.dictionaries?.carriers?.[carrierCode] || carrierCode;
            const departure = firstSegment.departure.at;
            const price = parseFloat(offer.price.total);
            // Create a unique key for this flight (Airline + Departure Time)
            const key = `${carrierCode}-${departure}`;
            // Only keep this flight if we haven't seen it yet, OR if this price is cheaper
            if (!uniqueFlights.has(key) || price < uniqueFlights.get(key).price) {
                uniqueFlights.set(key, {
                    airline: airline,
                    price: price,
                    currency: offer.price.currency,
                    departure: departure,
                    arrival: lastSegment.arrival.at,
                    duration: firstItinerary.duration.replace('PT', '').toLowerCase(),
                    stops: firstItinerary.segments.length - 1,
                    bookingLink: `Flight ID: ${offer.id}`
                });
            }
        });
        return Array.from(uniqueFlights.values()).slice(0, 10);
    }
    /**
     * Get airport or city suggestions (autocomplete)
     */
    async searchLocations(keyword) {
        if (!this.apiKey || this.apiKey === 'test_key_replace_later') {
            return [
                { code: 'JFK', name: 'John F. Kennedy International Airport', city: 'New York' },
                { code: 'LAX', name: 'Los Angeles International Airport', city: 'Los Angeles' },
                { code: 'ORD', name: "O'Hare International Airport", city: 'Chicago' },
            ];
        }
        try {
            const token = await this.getAccessToken();
            const response = await axios.get(`${this.baseUrl}/reference-data/locations`, {
                params: {
                    subType: 'AIRPORT,CITY',
                    keyword: keyword,
                    'page[limit]': 10,
                },
                headers: {
                    Authorization: `Bearer ${token}`,
                },
            });
            return response.data.data.map((location) => ({
                code: location.iataCode,
                name: location.name,
                city: location.address?.cityName || '',
                country: location.address?.countryName || '',
            }));
        }
        catch (error) {
            console.error('Location search error:', error);
            return [];
        }
    }
    /**
     * Return mock flight data for testing/fallback
     */
    getMockFlightData(params) {
        const basePrice = 350;
        const priceVariation = Math.floor(Math.random() * 200);
        return [
            {
                airline: 'United Airlines',
                price: basePrice + priceVariation,
                currency: 'USD',
                departure: `${params.departDate}T08:00:00`,
                arrival: `${params.departDate}T11:30:00`,
                duration: 'PT3H30M',
                stops: 0,
            },
            {
                airline: 'Delta Air Lines',
                price: basePrice + priceVariation - 50,
                currency: 'USD',
                departure: `${params.departDate}T10:15:00`,
                arrival: `${params.departDate}T15:45:00`,
                duration: 'PT5H30M',
                stops: 1,
            },
            {
                airline: 'American Airlines',
                price: basePrice + priceVariation + 100,
                currency: 'USD',
                departure: `${params.departDate}T06:30:00`,
                arrival: `${params.departDate}T09:45:00`,
                duration: 'PT3H15M',
                stops: 0,
            },
            {
                airline: 'Spirit Airlines',
                price: basePrice + priceVariation - 120,
                currency: 'USD',
                departure: `${params.departDate}T22:00:00`,
                arrival: `${params.departDate}T05:30:00`,
                duration: 'PT7H30M',
                stops: 1,
            },
        ];
    }
    /**
     * Format duration from ISO 8601 format to readable format
     */
    formatDuration(isoDuration) {
        const match = isoDuration.match(/PT(?:(\d+)H)?(?:(\d+)M)?/);
        if (!match)
            return isoDuration;
        const hours = match[1] ? `${match[1]}h` : '';
        const minutes = match[2] ? `${match[2]}m` : '';
        return `${hours}${hours && minutes ? ' ' : ''}${minutes}`;
    }
}
export default FlightService;
//# sourceMappingURL=FlightService.js.map