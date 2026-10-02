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
            const feature = geoResponse.data?.features?.[0];
            if (!feature) return { error: `Location "${location}" not found.`, results: [] };

            const { lon, lat } = feature.properties || {};
            const placesResponse = await axios.get(this.placesUrl, {
                params: {
                    categories: this.getCategoryKinds(category),
                    filter: `circle:${lon},${lat},${radius}`,
                    bias: `proximity:${lon},${lat}`,
                    limit: 50,
                    apiKey: this.apiKey,
                    lang: 'en'
                },
                timeout: 20000
            });

            const raw = [];
            for (const item of placesResponse.data?.features || []) {
                const p = item.properties || {};
                const name = String(p.name || '').trim();
                if (!name) continue;
                const categories = Array.isArray(p.categories) ? p.categories : [];

                raw.push({
                    placeId: p.place_id || null,
                    name,
                    description: p.description || null,
                    address: p.formatted || p.address_line2 || null,
                    coordinates: { lat: p.lat, lon: p.lon },
                    distanceMeters: Number.isFinite(p.distance) ? p.distance : null,
                    categories,
                    kinds: categories.join(',')
                });
            }

            return this.rankAndFilter(raw, category).slice(0, 8);
        } catch (error) {
            console.error('❌ Places API Error:', error.response?.data || error.message);
            return { error: `Live places search failed: ${error.message}`, results: [] };
        }
    }

    async getAttractions(city, limit = 8, radius = 10000) {
        const result = await this.searchPlaces(city, 'tourist_attractions', radius);
        return Array.isArray(result) ? result.slice(0, limit) : result;
    }

    async getRestaurants(location, limit = 8) {
        const result = await this.searchPlaces(location, 'restaurants', 5000);
        return Array.isArray(result) ? result.slice(0, limit) : result;
    }

    getCategoryKinds(category) {
        const map = {
            tourist_attractions: [
                'tourism.attraction',
                'tourism.sights',
                'entertainment.museum',
                'entertainment.culture',
                'building.place_of_worship',
                'leisure.park',
                'natural'
            ].join(','),
            restaurants: 'catering.restaurant,catering.cafe,catering.fast_food',
            hotels: 'accommodation.hotel,accommodation.guest_house',
            entertainment: 'entertainment',
            nature: 'natural,leisure.park,leisure.park.garden',
            shopping: 'commercial.shopping_mall,commercial.marketplace',
            religion: 'building.place_of_worship,tourism.sights.place_of_worship'
        };
        return map[category || 'tourist_attractions'] || map.tourist_attractions;
    }

    rankAndFilter(places, category) {
        const infrastructure = /(road|street|highway|lane|path|way|junction|roundabout|signal|bus stop|bus station|railway|station|parking|building|mawatha)$/i;
        const weakAttraction = /(statue|viewpoint|triangle|building|road|street|path|train|water works|waterworks|water supply|car shelter|holy car|yaanaikal|nandi|pillar|pillars|thungal|memorial|monument)$/i;
        const restaurantNameProblem = /(street|road|lane|mawatha|marg|highway|junction|bus stop|station)$/i;

        const filtered = places.filter(place => {
            const categories = place.categories || [];
            const categoryText = categories.join(',').toLowerCase();
            const name = place.name.trim();

            if (category === 'restaurants') {
                const isFoodCategory = categories.some(c =>
                    c.startsWith('catering.restaurant') ||
                    c.startsWith('catering.cafe') ||
                    c.startsWith('catering.fast_food')
                );
                if (!isFoodCategory) return false;
                if (restaurantNameProblem.test(name)) return false;
                return true;
            }

            if (category !== 'tourist_attractions') return true;

            const isStrongTourism = categories.some(c => (
                c === 'tourism.attraction' ||
                c === 'tourism.sights' ||
                c.includes('museum') ||
                c.includes('culture') ||
                c.includes('place_of_worship') ||
                c.includes('park') ||
                c.startsWith('natural')
            ));
            if (!isStrongTourism) return false;

            // Do not let map infrastructure masquerade as a tourist attraction.
            if (infrastructure.test(name)) return false;

            // These are usually map-level objects or small monuments rather than
            // destination-level attractions. Keep them out of the main itinerary.
            // A city with fewer good results is preferable to fabricated/padded
            // sightseeing recommendations.
            if (weakAttraction.test(name)) {
                const legitimateContext =
                    categoryText.includes('museum') ||
                    categoryText.includes('culture') ||
                    categoryText.includes('place_of_worship') ||
                    categoryText.includes('historic');
                if (!legitimateContext) return false;
            }

            return true;
        });

        const scored = filtered.map(place => {
            let score = 0;
            const categories = place.categories || [];

            if (category === 'tourist_attractions') {
                if (categories.includes('tourism.attraction')) score += 120;
                if (categories.includes('tourism.sights')) score += 110;
                if (categories.some(c => c.includes('museum'))) score += 105;
                if (categories.some(c => c.includes('culture'))) score += 100;
                if (categories.some(c => c.includes('place_of_worship'))) score += 98;
                if (categories.some(c => c.includes('historic'))) score += 96;
                if (categories.some(c => c.includes('park'))) score += 80;
                if (categories.some(c => c.startsWith('natural'))) score += 75;
            } else if (category === 'restaurants') {
                if (categories.includes('catering.restaurant')) score += 100;
                if (categories.includes('catering.cafe')) score += 80;
                if (categories.includes('catering.fast_food')) score += 60;
            }

            if (Number.isFinite(place.distanceMeters)) {
                score += Math.max(0, 25 - place.distanceMeters / 500);
            }

            return { ...place, _score: score };
        });

        scored.sort((a, b) => b._score - a._score);

        const seen = new Set();
        return scored.filter(place => {
            const key = place.name.toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
            if (seen.has(key)) return false;
            seen.add(key);
            delete place._score;
            return true;
        });
    }
}

export default PlacesService;
