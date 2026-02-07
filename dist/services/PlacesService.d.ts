export interface Place {
    name: string;
    description: string;
    address: string;
    coordinates: {
        lat: number;
        lon: number;
    };
    kinds: string;
    categories?: string;
    rating?: number;
    image?: string;
}
export declare class PlacesService {
    private apiKey;
    private geocodeUrl;
    private placesUrl;
    constructor();
    /**
     * Search for places, attractions, and points of interest
     */
    searchPlaces(location: string, category?: string, radius?: number): Promise<Place[]>;
    /**
     * Get tourist attractions in a city
     */
    getAttractions(city: string, limit?: number): Promise<Place[]>;
    /**
     * Get restaurants in a location
     */
    getRestaurants(location: string, limit?: number): Promise<Place[]>;
    /**
     * Map generic categories to Geoapify specific categories
     * Reference: https://apidocs.geoapify.com/docs/places/#categories
     */
    private getCategoryKinds;
    /**
     * Return mock data when API key is not available
     */
    private getMockPlaces;
}
export default PlacesService;
//# sourceMappingURL=PlacesService.d.ts.map