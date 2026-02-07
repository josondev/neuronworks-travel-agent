export declare class WeatherService {
    private apiKey;
    private baseUrl;
    constructor();
    getWeatherForecast(args: any): Promise<{
        date: string;
        temperature: number;
        description: string;
    }[] | {
        date: any;
        temperature: number;
        description: any;
        humidity: any;
        windSpeed: any;
    }[]>;
    private getMockWeather;
}
//# sourceMappingURL=WeatherService.d.ts.map