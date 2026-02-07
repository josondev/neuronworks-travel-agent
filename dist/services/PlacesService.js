import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();
export class PlacesService {
    apiKey;
    // Geoapify separates Geocoding and Places into different endpoints
    geocodeUrl = 'https://api.geoapify.com/v1/geocode/search';
    placesUrl = 'https://api.geoapify.com/v2/places';
    constructor() {
        this.apiKey = process.env.GEOAPIFY_API_KEY || '';
        if (!this.apiKey) {
            console.warn('Geoapify API key not found. Places service will return mock data.');
        }
    }
    /**
     * Search for places, attractions, and points of interest
     */
    async searchPlaces(location, category, radius = 5000) {
        try {
            if (!this.apiKey || this.apiKey === 'test_key_replace_later') {
                return this.getMockPlaces(location, category);
            }
            // 1. Get coordinates for the location using Geoapify Geocoding API
            const geoResponse = await axios.get(this.geocodeUrl, {
                params: {
                    text: location,
                    apiKey: this.apiKey,
                    limit: 1
                }
            });
            if (!geoResponse.data || !geoResponse.data.features || geoResponse.data.features.length === 0) {
                throw new Error(`Location "${location}" not found`);
            }
            // Geoapify returns GeoJSON
            const locationFeature = geoResponse.data.features[0];
            const lon = locationFeature.properties.lon;
            const lat = locationFeature.properties.lat;
            // 2. Search for places around those coordinates using Geoapify Places API
            const placesResponse = await axios.get(this.placesUrl, {
                params: {
                    categories: this.getCategoryKinds(category),
                    filter: `circle:${lon},${lat},${radius}`,
                    limit: 10,
                    apiKey: this.apiKey,
                    lang: 'en'
                }
            });
            const places = [];
            // 3. Map Geoapify results to our Place interface
            for (const item of placesResponse.data.features || []) {
                const props = item.properties;
                // Skip items without names (often generic map points)
                if (!props.name && !props.formatted)
                    continue;
                places.push({
                    name: props.name || props.address_line1 || 'Unknown Place',
                    // Geoapify doesn't provide long descriptions, so we construct one from categories or address
                    description: props.formatted || `A ${props.categories?.join(', ')} located in ${props.city || 'the area'}`,
                    address: props.formatted || props.address_line2 || 'Address not available',
                    coordinates: {
                        lat: props.lat,
                        lon: props.lon
                    },
                    // Convert Geoapify category array to comma-separated string
                    kinds: props.categories ? props.categories.join(',') : 'interesting_places',
                    categories: props.categories ? props.categories.join(',') : undefined,
                    // Geoapify doesn't standardly provide ratings/images in the base plan
                    // We leave these undefined or null unless extended data is available
                    rating: undefined,
                    image: undefined
                });
            }
            return places;
        }
        catch (error) {
            console.error('Error searching places:', error);
            return this.getMockPlaces(location, category);
        }
    }
    /**
     * Get tourist attractions in a city
     */
    async getAttractions(city, limit = 5) {
        return this.searchPlaces(city, 'tourist_attractions', 10000);
    }
    /**
     * Get restaurants in a location
     */
    async getRestaurants(location, limit = 5) {
        return this.searchPlaces(location, 'restaurants', 5000);
    }
    /**
     * Map generic categories to Geoapify specific categories
     * Reference: https://apidocs.geoapify.com/docs/places/#categories
     */
    getCategoryKinds(category) {
        const categoryMap = {
            'tourist_attractions': 'tourism,entertainment.culture,building.historic',
            'restaurants': 'catering.restaurant',
            'hotels': 'accommodation',
            'entertainment': 'entertainment,leisure',
            'nature': 'natural,leisure.park,beach',
            'shopping': 'commercial.shopping_mall,commercial.supermarket',
            'religion': 'building.place_of_worship'
        };
        return categoryMap[category || 'tourist_attractions'] || 'tourism';
    }
    /**
     * Return mock data when API key is not available
     */
    getMockPlaces(location, category) {
        const categoryName = category || 'attractions';
        return [
            {
                name: `Popular ${categoryName} in ${location} #1`,
                description: `This is a highly rated ${categoryName} location in ${location}. A must-visit destination with excellent reviews.`,
                address: `123 Main Street, ${location}`,
                coordinates: { lat: 40.7128, lon: -74.0060 },
                kinds: 'tourist_attractions',
                rating: 4.5
            },
            {
                name: `Historic ${categoryName} in ${location}`,
                description: `A historic landmark in ${location} with rich cultural significance and beautiful architecture.`,
                address: `456 Heritage Ave, ${location}`,
                coordinates: { lat: 40.7589, lon: -73.9851 },
                kinds: 'historic,cultural',
                rating: 4.7
            },
            {
                name: `Modern ${categoryName} Center`,
                description: `Contemporary ${categoryName} venue in ${location} offering unique experiences and modern facilities.`,
                address: `789 Innovation Blvd, ${location}`,
                coordinates: { lat: 40.7614, lon: -73.9776 },
                kinds: 'entertainment',
                rating: 4.3
            },
            {
                name: `${location} Cultural District`,
                description: `Vibrant cultural area featuring multiple ${categoryName} options and local experiences.`,
                address: `321 Culture Street, ${location}`,
                coordinates: { lat: 40.7484, lon: -73.9857 },
                kinds: 'cultural,museums',
                rating: 4.6
            },
            {
                name: `Scenic ${categoryName} Spot`,
                description: `Beautiful ${categoryName} location in ${location} known for stunning views and peaceful atmosphere.`,
                address: `654 Scenic View, ${location}`,
                coordinates: { lat: 40.7829, lon: -73.9654 },
                kinds: 'natural,tourist_attractions',
                rating: 4.8
            }
        ];
    }
}
export default PlacesService;
//# sourceMappingURL=PlacesService.js.map
