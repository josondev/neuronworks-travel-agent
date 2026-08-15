import axios from 'axios';
import dotenv from 'dotenv';
dotenv.config();

export class WeatherService {
    constructor() {
        this.apiKey = process.env.OPENWEATHER_API_KEY || '';
        this.baseUrl = 'https://api.openweathermap.org/data/2.5';
    }

    async getWeatherForecast(args) {
        if (!this.apiKey) {
            return { error: 'Live weather unavailable: OPENWEATHER_API_KEY is not configured.', results: [] };
        }

        try {
            const geoResponse = await axios.get('https://api.openweathermap.org/geo/1.0/direct', {
                params: { q: args.city, limit: 1, appid: this.apiKey },
                timeout: 15000
            });

            if (!geoResponse.data?.length) {
                return { error: `City not found: ${args.city}`, results: [] };
            }

            const { lat, lon } = geoResponse.data[0];
            const response = await axios.get(`${this.baseUrl}/forecast`, {
                params: { lat, lon, units: 'metric', appid: this.apiKey },
                timeout: 15000
            });

            const requestedStart = args.startDate;
            const requestedEnd = args.endDate;
            const dailyForecasts = [];
            const seenDates = new Set();

            for (const item of response.data?.list || []) {
                const date = item.dt_txt?.split(' ')[0];
                const time = item.dt_txt?.split(' ')[1];
                if (!date || date < requestedStart || date > requestedEnd || seenDates.has(date)) continue;
                if (!time?.startsWith('12:')) continue;

                dailyForecasts.push({
                    date,
                    temperature: Math.round(item.main.temp),
                    description: item.weather?.[0]?.description || null,
                    humidity: item.main.humidity,
                    windSpeed: item.wind?.speed
                });
                seenDates.add(date);
            }

            if (dailyForecasts.length === 0) {
                return {
                    error: 'No live forecast is available for the requested date range from the current forecast endpoint.',
                    results: []
                };
            }

            return dailyForecasts;
        } catch (error) {
            console.error('❌ Weather API Error:', error.response?.data || error.message);
            return { error: `Live weather request failed: ${error.message}`, results: [] };
        }
    }
}
