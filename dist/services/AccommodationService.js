import axios from 'axios';
export class AccommodationService {
    apiKey;
    baseUrl = 'https://api.geoapify.com/v2/places';
    geocodeUrl = 'https://api.geoapify.com/v1/geocode/search';
    constructor() {
        this.apiKey = process.env.GEOAPIFY_API_KEY || '';
    }
    async searchAccommodation(args) {
        if (!this.apiKey)
            return this.getMockHotels(args.city);
        try {
            // 1. Geocode the city
            const geoResponse = await axios.get(this.geocodeUrl, {
                params: { text: args.city, apiKey: this.apiKey }
            });
            if (!geoResponse.data.features?.length)
                return this.getMockHotels(args.city);
            const { lat, lon } = geoResponse.data.features[0].properties;
            // 2. Search for real hotels in that city
            const placesResponse = await axios.get(this.baseUrl, {
                params: {
                    categories: 'accommodation.hotel',
                    filter: `circle:${lon},${lat},5000`, // 5km radius
                    limit: 10,
                    apiKey: this.apiKey
                }
            });
            // 3. Transform to "Booking" format with ESTIMATED prices
            // (Geoapify doesn't give rates, so we estimate based on the hotel "feel")
            return placesResponse.data.features.map((feature) => {
                const isFancy = feature.properties.name?.toLowerCase().includes('grand') ||
                    feature.properties.name?.toLowerCase().includes('luxury') ||
                    feature.properties.name?.toLowerCase().includes('plaza');
                const basePrice = isFancy ? 250 : 80;
                const randomVar = Math.floor(Math.random() * 40);
                return {
                    name: feature.properties.name || "Unknown Hotel",
                    price: `${basePrice + randomVar} USD`,
                    rating: isFancy ? 4.8 : 4.2,
                    address: feature.properties.address_line2 || feature.properties.formatted,
                    website: feature.properties.website || "Not available",
                    source: "Real Location (Price Estimated)"
                };
            });
        }
        catch (error) {
            console.error('Accommodation Error:', error);
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
//# sourceMappingURL=AccommodationService.js.map