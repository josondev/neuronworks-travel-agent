export interface FlightSearchParams {
    origin: string;
    destination: string;
    departDate: string;
    returnDate?: string;
    passengers: number;
    travelClass?: 'ECONOMY' | 'PREMIUM_ECONOMY' | 'BUSINESS' | 'FIRST';
}
export interface FlightResult {
    airline: string;
    price: number;
    currency: string;
    departure: string;
    arrival: string;
    duration: string;
    stops: number;
    bookingLink?: string;
}
export declare class FlightService {
    private apiKey;
    private apiSecret;
    private baseUrl;
    private accessToken;
    private tokenExpiry;
    constructor();
    /**
     * Get OAuth access token from Amadeus
     */
    private getAccessToken;
    /**
     * Search for flights using Amadeus Flight Offers Search API
     */
    searchFlights(params: FlightSearchParams): Promise<FlightResult[]>;
    /**
     * Transform Amadeus API response to our FlightResult format
     */
    /**
     * Transform Amadeus API response to our FlightResult format
     * AND remove duplicates
     */
    private transformAmadeusResponse;
    /**
     * Get airport or city suggestions (autocomplete)
     */
    searchLocations(keyword: string): Promise<any[]>;
    /**
     * Return mock flight data for testing/fallback
     */
    private getMockFlightData;
    /**
     * Format duration from ISO 8601 format to readable format
     */
    formatDuration(isoDuration: string): string;
}
export default FlightService;
//# sourceMappingURL=FlightService.d.ts.map