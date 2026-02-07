import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

export class FlightService {
    constructor() {
        this.apiKey = process.env.AMADEUS_CLIENT_ID || '';
        this.apiSecret = process.env.AMADEUS_CLIENT_SECRET || '';
        this.baseUrl = 'https://test.api.amadeus.com/v2';
        this.accessToken = '';
        this.tokenExpiry = 0;

        if (!this.apiKey || this.apiKey === 'test_key_replace_later') {
            console.warn('⚠️ Amadeus API key not configured. Flight service will return mock data.');
        }
    }

    async getAccessToken() {
        if (this.accessToken && Date.now() < this.tokenExpiry) {
            return this.accessToken;
        }
        try {
            const response = await axios.post('https://test.api.amadeus.com/v1/security/oauth2/token', 
                new URLSearchParams({
                    grant_type: 'client_credentials',
                    client_id: this.apiKey,
                    client_secret: this.apiSecret,
                }), {
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                }
            );
            this.accessToken = response.data.access_token;
            this.tokenExpiry = Date.now() + (response.data.expires_in - 300) * 1000;
            return this.accessToken;
        } catch (error) {
            console.error('❌ Failed to get Amadeus access token:', error.message);
            throw new Error('Authentication failed with Amadeus API');
        }
    }

    async searchFlights(params) {
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
                max: 10
            };

            if (params.returnDate) searchParams.returnDate = params.returnDate;

            const response = await axios.get(`${this.baseUrl}/shopping/flight-offers`, {
                params: searchParams,
                headers: { Authorization: `Bearer ${token}` },
            });

            return this.transformAmadeusResponse(response.data);
        } catch (error) {
            console.error('❌ Amadeus API error:', error.response?.data || error.message);
            return this.getMockFlightData(params);
        }
    }

    transformAmadeusResponse(data) {
        if (!data.data || data.data.length === 0) return [];
        
        const uniqueFlights = new Map();
        data.data.forEach((offer) => {
            const firstItinerary = offer.itineraries[0];
            const firstSegment = firstItinerary.segments[0];
            const lastSegment = firstItinerary.segments[firstItinerary.segments.length - 1];
            const carrierCode = firstSegment.carrierCode;
            const airline = data.dictionaries?.carriers?.[carrierCode] || carrierCode;
            const departure = firstSegment.departure.at;
            const price = parseFloat(offer.price.total);
            
            const key = `${carrierCode}-${departure}`;
            
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
                duration: '3h 30m',
                stops: 0,
            },
            {
                airline: 'Delta Air Lines',
                price: basePrice + priceVariation - 50,
                currency: 'USD',
                departure: `${params.departDate}T10:15:00`,
                arrival: `${params.departDate}T15:45:00`,
                duration: '5h 30m',
                stops: 1,
            }
        ];
    }
}