export declare class CurrencyService {
    private apiKey;
    private baseUrl;
    constructor();
    getExchangeRate(args: {
        from: string;
        to: string;
        amount?: number;
    }): Promise<{
        from: string;
        to: string;
        amount: number;
        rate: number;
        convertedAmount: number;
        note: string;
        lastUpdate?: undefined;
        error?: undefined;
    } | {
        from: string;
        to: string;
        amount: number;
        rate: any;
        convertedAmount: any;
        lastUpdate: any;
        note?: undefined;
        error?: undefined;
    } | {
        error: string;
        from?: undefined;
        to?: undefined;
        amount?: undefined;
        rate?: undefined;
        convertedAmount?: undefined;
        note?: undefined;
        lastUpdate?: undefined;
    }>;
}
//# sourceMappingURL=CurrencyService.d.ts.map