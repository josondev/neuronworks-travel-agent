export declare class AccommodationService {
    private apiKey;
    private baseUrl;
    private geocodeUrl;
    constructor();
    searchAccommodation(args: {
        city: string;
        checkIn: string;
        checkOut: string;
        guests: number;
    }): Promise<any>;
    private getMockHotels;
}
//# sourceMappingURL=AccommodationService.d.ts.map