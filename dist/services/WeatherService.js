import axios from 'axios';
import dotenv from 'dotenv';
dotenv.config();

export class WeatherService {
    constructor() {
        this.apiKey = process.env.OPENWEATHER_API_KEY || '';
        this.baseUrl = 'https://api.openweathermap.org/data/2.5';
    }

    async getWeatherForecast(args) {
        if (!this.apiKey) return this.getMockWeather(args.city);
        
        try {
            // 1. Get coordinates
            const geoUrl = `http://api.openweathermap.org/geo/1.0/direct?q=${args.city}&limit=1&appid=${this.apiKey}`;
            const geoRes = await axios.get(geoUrl);
            
            if (!geoRes.data.length) throw new Error('City not found');
            
            const { lat, lon } = geoRes.data[0];

            // 2. Get Forecast
            const weatherUrl = `${this.baseUrl}/forecast?lat=${lat}&lon=${lon}&units=metric&appid=${this.apiKey}`;
            const response = await axios.get(weatherUrl);

            // 3. Filter: Take one reading per day (noon)
            const dailyForecasts = [];
            const seenDates = new Set();
            
            for (const item of response.data.list) {
                const date = item.dt_txt.split(' ')[0];
                const time = item.dt_txt.split(' ')[1];
                
                if (!seenDates.has(date) && time.includes('12:00')) {
                    dailyForecasts.push({
                        date: date,
                        temperature: Math.round(item.main.temp),
                        description: item.weather[0].description,
                        humidity: item.main.humidity,
                        windSpeed: item.wind.speed
                    });
                    seenDates.add(date);
                }
            }
            return dailyForecasts.slice(0, 5);
        } catch (error) {
            console.error('❌ Weather API Error:', error.message);
            return this.getMockWeather(args.city);
        }
    }

    getMockWeather(city) {
        return Array.from({ length: 5 }).map((_, i) => ({
            date: `2025-06-0${i + 1}`,
            temperature: 25,
            description: "sunny"
        }));
    }
}