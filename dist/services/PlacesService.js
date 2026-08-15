import axios from 'axios';

export class PlacesService {
    constructor() {
        this.apiKey = process.env.GEOAPIFY_API_KEY || '';
        this.geocodeUrl = 'https://api.geoapify.com/v1/geocode/search';
        this.placesUrl = 'https://api.geoapify.com/v2/places';
    }

    async searchPlaces(location, category, radius = 5000) {
        if (!this.apiKey || this.apiKey === 'test_key_replace_later') {
            return { error: 'Live places search unavailable: GEOAPIFY_API_KEY is not configured.', results: [] };
        }

        if (!location || typeof location !== 'string') {
            return { error: 'A location is required.', results: [] };
        }

        try {
            const geoResponse = await axios.get(this.geocodeUrl, {
                params: { text: location, apiKey: this.apiKey, limit: 1 },
                timeout: 15000
            });

            const features = geoResponse.data?.features || [];
            if (features.length === 0) {
                return { error: `Location "${location}" not found.`, results: [] };
            }

            const props = features[0].properties;
            const lon = props.lon;
            const lat = props.lat;

            const placesResponse = await axios.get(this.placesUrl, {
                params: {
                    categories: this.getCategoryKinds(category),
                    filter: `circle:${lon},${lat},${radius}`,
                    limit: 10,
                    apiKey: this.apiKey,
                    lang: 'en'
                },
                timeout: 20000
            });

            const places = [];
            for (const item of placesResponse.data?.features || []) {
                const place = item.properties || {};
                if (!place.name && !place.formatted) continue;

                places.push({
                    name: place.name || place.address_line1 || 'Unknown Place',
                    description: place.formatted || null,
                    address: place.formatted || place.address_line2 || null,
                    coordinates: { lat: place.lat, lon: place.lon },
                    kinds: Array.isArray(place.categories) ? place.categories.join(',') : null,
                    categories: Array.isArray(place.categories) ? place.categories : []
                });
            }

            return places;
        } catch (error) {
            console.error('❌ Places API Error:', error.response?.data || error.message);
            return { error: `Live places search failed: ${error.message}`, results: [] };
        }
    }

    async getAttractions(city, limit = 5) {
        const result = await this.searchPlaces(city, 'tourist_attractions', 10000);
        return Array.isArray(result) ? result.slice(0, limit) : result;
    }

    async getRestaurants(location, limit = 5) {
        const result = await this.searchPlaces(location, 'restaurants', 5000);
        return Array.isArray(result) ? result.slice(0, limit) : result;
    }

    getCategoryKinds(category) {
        const categoryMap = {
            tourist_attractions: 'tourism,entertainment.culture,building.historic',
            restaurants: 'catering.restaurant',
            hotels: 'accommodation',
            entertainment: 'entertainment,leisure',
            nature: 'natural,leisure.park,beach',
            shopping: 'commercial.shopping_mall,commercial.supermarket',
            religion: 'building.place_of_worship'
        };
        return categoryMap[category || 'tourist_attractions'] || 'tourism';
    }
}

export default PlacesService;
