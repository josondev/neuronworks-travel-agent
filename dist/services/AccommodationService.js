import axios from 'axios';

export class AccommodationService {
    constructor() {
        this.apiKey = process.env.SERPAPI_API_KEY || '';
        this.baseUrl = 'https://serpapi.com/search.json';
    }

    async searchAccommodation(args) {
        const city = String(args?.city || '').trim();
        const checkIn = String(args?.checkIn || '').trim();
        const checkOut = String(args?.checkOut || '').trim();
        const adults = Math.max(1, Number(args?.adults) || 1);

        if (!this.apiKey) {
            console.error('❌ Hotel search unavailable: SERPAPI_API_KEY is missing.');
            return {
                error: 'Live hotel search unavailable because SERPAPI_API_KEY is not configured.',
                results: []
            };
        }

        if (!city || !checkIn || !checkOut) {
            return {
                error: 'Live hotel pricing requires city, checkIn, and checkOut dates.',
                results: []
            };
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
                    amenities: Array.isArray(property.amenities)
                        ? property.amenities.slice(0, 12)
                        : [],
                    propertyToken: property.property_token || null,
                    searchLink: property.link || property.serpapi_property_details_link || null,
                    source: 'Google Hotels via SerpApi'
                }))
                .filter((hotel) => hotel.name && hotel.price !== null);

            return results.slice(0, 10);
        } catch (error) {
            console.error(
                '❌ Google Hotels / SerpApi error:',
                error.response?.data || error.message
            );
            return {
                error: `Live hotel search failed: ${error.message}`,
                results: []
            };
        }
    }
}
