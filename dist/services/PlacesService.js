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
            if (!features.length) {
                return { error: `Location "${location}" not found.`, results: [] };
            }

            const props = features[0].properties || {};
            const lon = props.lon;
            const lat = props.lat;

            const placesResponse = await axios.get(this.placesUrl, {
                params: {
                    categories: this.getCategoryKinds(category),
                    filter: `circle:${lon},${lat},${radius}`,
                    bias: `proximity:${lon},${lat}`,
                    limit: 30,
                    apiKey: this.apiKey,
                    lang: 'en'
                },
                timeout: 20000
            });

            const raw = [];
            for (const item of placesResponse.data?.features || []) {
                const place = item.properties || {};
                const name = String(place.name || place.address_line1 || '').trim();
                if (!name) continue;

                const categories = Array.isArray(place.categories) ? place.categories : [];
                const categoryText = categories.join(',');
                raw.push({
                    name,
                    description: place.description || place.formatted || null,
                    address: place.formatted || place.address_line2 || null,
                    coordinates: { lat: place.lat, lon: place.lon },
                    distanceMeters: place.distance ?? null,
                    categories,
                    kinds: categoryText
                });
            }

            return this.rankAndFilter(raw, category).slice(0, category === 'tourist_attractions' ? 12 : 12);
        } catch (error) {
            console.error('❌ Places API Error:', error.response?.data || error.message);
            return { error: `Live places search failed: ${error.message}`, results: [] };
        }
    }

    async getAttractions(city, limit = 10, radius = 10000) {
        const result = await this.searchPlaces(city, 'tourist_attractions', radius);
        return Array.isArray(result) ? result.slice(0, limit) : result;
    }

    async getRestaurants(location, limit = 10) {
        const result = await this.searchPlaces(location, 'restaurants', 5000);
        return Array.isArray(result) ? result.slice(0, limit) : result;
    }

    getCategoryKinds(category) {
        const categoryMap = {
            // Broad 'tourism' results were returning roads and miscellaneous infrastructure.
            // These more specific Geoapify categories are intended for actual tourist POIs.
            tourist_attractions: [
                'tourism.attraction',
                'tourism.sights',
                'entertainment.museum',
                'entertainment.culture',
                'leisure.park',
                'natural'
            ].join(','),
            restaurants: 'catering.restaurant,catering.cafe',
            hotels: 'accommodation.hotel,accommodation.guest_house',
            entertainment: 'entertainment',
            nature: 'natural,leisure.park,leisure.park.garden',
            shopping: 'commercial.shopping_mall,commercial.marketplace,commercial.supermarket',
            religion: 'tourism.sights.place_of_worship,building.place_of_worship'
        };
        return categoryMap[category || 'tourist_attractions'] || 'tourism.attraction,tourism.sights';
    }

    rankAndFilter(places, category) {
        const bannedForAttractions = [
            'road',
            'street',
            'main road',
            'highway',
            'junction',
            'bus stop',
            'bus station',
            'railway',
            'station',
            'parking',
            'roundabout',
            'signal',
            'traffic'
        ];

        const scored = places
            .filter((place) => {
                if (category !== 'tourist_attractions') return true;
                const text = `${place.name} ${place.description || ''} ${place.kinds || ''}`.toLowerCase();
                // Exclude infrastructure unless it is explicitly tagged as a tourism sight.
                const hasTourismTag = place.categories.some(c => c.startsWith('tourism.'));
                if (!hasTourismTag && bannedForAttractions.some(term => text.includes(term))) return false;
                return true;
            })
            .map((place) => {
                let score = 0;
                const categories = place.categories || [];

                if (category === 'tourist_attractions') {
                    if (categories.some(c => c === 'tourism.sights')) score += 100;
                    if (categories.some(c => c === 'tourism.attraction')) score += 95;
                    if (categories.some(c => c.includes('place_of_worship'))) score += 90;
                    if (categories.some(c => c.includes('museum') || c.includes('culture'))) score += 85;
                    if (categories.some(c => c.includes('park') || c.startsWith('natural'))) score += 75;
                    if (categories.some(c => c.includes('statue') || c.includes('memorial'))) score -= 20;
                }

                if (category === 'restaurants') {
                    if (categories.includes('catering.restaurant')) score += 100;
                    if (categories.includes('catering.cafe')) score += 70;
                }

                if (Number.isFinite(place.distanceMeters)) {
                    score += Math.max(0, 30 - place.distanceMeters / 500);
                }

                return { ...place, _score: score };
            })
            .sort((a, b) => b._score - a._score)
            .map(({ _score, ...place }) => place);

        // Deduplicate by normalized name.
        const seen = new Set();
        return scored.filter((place) => {
            const key = place.name.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
        });
    }
}

export default PlacesService;
