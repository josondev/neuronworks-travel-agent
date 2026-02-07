import axios from 'axios';

export class AccommodationService {
    constructor() {
        this.apiKey = process.env.GEOAPIFY_KEY || '';
        this.baseUrl = 'https://api.geoapify.com/v2/places';
        this.geocodeUrl = 'https://api.geoapify.com/v1/geocode/search';
    }

    async searchAccommodation(args) {
        if (!this.apiKey) return this.getMockHotels(args.city);
        
        try {
            // 1. Geocode the city
            const geoResponse = await axios.get(this.geocodeUrl, {
                params: { text: args.city, apiKey: this.apiKey }
            });

            if (!geoResponse.data.features?.length) {
                return this.getMockHotels(args.city);
            }

            const { lat, lon } = geoResponse.data.features[0].properties;

            // 2. Search for hotels
            const placesResponse = await axios.get(this.baseUrl, {
                params: {
                    categories: 'accommodation.hotel',
                    filter: `circle:${lon},${lat},5000`,
                    limit: 10,
                    apiKey: this.apiKey
                }
            });

            // 3. Transform Data
            return placesResponse.data.features.map((feature) => {
                const isFancy = feature.properties.name?.toLowerCase().includes('grand') ||
                                feature.properties.name?.toLowerCase().includes('luxury');
                const basePrice = isFancy ? 250 : 80;
                
                return {
                    name: feature.properties.name || "Unknown Hotel",
                    price: `${basePrice + Math.floor(Math.random() * 40)} USD`,
                    rating: isFancy ? 4.8 : 4.2,
                    address: feature.properties.address_line2 || feature.properties.formatted,
                    website: feature.properties.website || "Not available"
                };
            });
        } catch (error) {
            console.error('❌ Accommodation Error:', error.message);
            return this.getMockHotels(args.city);
        }
    }

    getMockHotels(city) {
        return [
            { name: `Grand Hotel ${city}`, price: "150 USD", rating: 8.5, address: "Downtown" },
            { name: `${city} City Hostel`, price: "45 USD", rating: 7.2, address: "Old Town" },
        ];
    }
}