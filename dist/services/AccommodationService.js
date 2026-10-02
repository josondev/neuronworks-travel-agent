import axios from 'axios';
import dotenv from 'dotenv';
dotenv.config();

export class AccommodationService {
    constructor() {
        this.apiKey = process.env.SERPAPI_API_KEY || process.env.SERPAPI_KEY || '';
        this.baseUrl = 'https://serpapi.com/search.json';
    }

    async searchAccommodation(args = {}) {
        const city = String(args.city || '').trim();
        const checkIn = String(args.checkIn || '').trim();
        const checkOut = String(args.checkOut || '').trim();
        const adults = Math.max(1, Number(args.adults) || 1);

        if (!this.apiKey) {
            return {
                error: 'Hotel search unavailable: SERPAPI_API_KEY (or SERPAPI_KEY) is not configured on the server.',
                results: []
            };
        }

        if (!city || !/^\d{4}-\d{2}-\d{2}$/.test(checkIn) || !/^\d{4}-\d{2}-\d{2}$/.test(checkOut)) {
            return {
                error: 'Hotel search requires city, checkIn and checkOut in YYYY-MM-DD format.',
                results: []
            };
        }

        if (new Date(`${checkOut}T00:00:00Z`) <= new Date(`${checkIn}T00:00:00Z`)) {
            return { error: 'Hotel checkOut must be after checkIn.', results: [] };
        }

        try {
            const response = await axios.get(this.baseUrl, {
                params: {
                    engine: 'google_hotels',
                    api_key: this.apiKey,
                    q: city,
                    check_in_date: checkIn,
                    check_out_date: checkOut,
                    adults,
                    sort_by: 3,
                    currency: 'USD',
                    gl: process.env.SERPAPI_GL || 'in',
                    hl: 'en'
                },
                timeout: 30000
            });

            if (response.data?.error) throw new Error(response.data.error);

            const properties = Array.isArray(response.data?.properties)
                ? response.data.properties
                : [];

            // Do not throw away a real hotel merely because SerpApi did not expose
            // an extracted price for that property. Preserve the provider data and
            // let the UI/model say "price unavailable" honestly.
            const results = properties
                .map((property) => ({
                    name: property.name || 'Unknown hotel',
                    price: Number.isFinite(property.rate_per_night?.extracted_lowest)
                        ? property.rate_per_night.extracted_lowest
                        : null,
                    priceBeforeTaxesFees: Number.isFinite(property.rate_per_night?.extracted_before_taxes_fees)
                        ? property.rate_per_night.extracted_before_taxes_fees
                        : null,
                    currency: response.data?.search_parameters?.currency || 'USD',
                    rating: Number.isFinite(property.overall_rating) ? property.overall_rating : null,
                    reviews: Number.isFinite(property.reviews) ? property.reviews : null,
                    hotelClass: property.hotel_class ?? null,
                    address: property.address || property.description || null,
                    amenities: Array.isArray(property.amenities) ? property.amenities.slice(0, 12) : [],
                    propertyToken: property.property_token || null,
                    searchLink: property.link || property.serpapi_property_details_link || null,
                    source: 'Google Hotels via SerpApi'
                }))
                .filter(hotel => hotel.name);

            if (!results.length) {
                return {
                    error: `No live Google Hotels properties were returned for ${city} from ${checkIn} to ${checkOut}.`,
                    results: [],
                    searchLink: response.data?.search_metadata?.google_hotels_url || null,
                    source: 'Google Hotels via SerpApi'
                };
            }

            return results.slice(0, 10);
        } catch (error) {
            const providerError = error.response?.data?.error || error.response?.data?.message || error.message;
            console.error('❌ Google Hotels / SerpApi error:', error.response?.data || error.message);
            return {
                error: `Live hotel search failed: ${providerError}`,
                results: []
            };
        }
    }
}
