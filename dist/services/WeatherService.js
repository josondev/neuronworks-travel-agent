import axios from 'axios';
import dotenv from 'dotenv';
dotenv.config();

export class WeatherService {
    constructor() {
        this.apiKey = process.env.OPENWEATHER_API_KEY || process.env.OPENWEATHERMAP_API_KEY || '';
        this.baseUrl = 'https://api.openweathermap.org/data/2.5';
    }

    async getWeatherForecast(args = {}) {
        const city = String(args.city || '').trim();
        const country = String(args.country || '').trim();
        const requestedStart = String(args.startDate || '').trim();
        const requestedEnd = String(args.endDate || '').trim();

        if (!this.apiKey) {
            return {
                error: 'Weather unavailable: OPENWEATHER_API_KEY (or OPENWEATHERMAP_API_KEY) is not configured on the server.',
                results: []
            };
        }

        if (!city || !/^\d{4}-\d{2}-\d{2}$/.test(requestedStart) || !/^\d{4}-\d{2}-\d{2}$/.test(requestedEnd)) {
            return {
                error: 'Weather requires city and startDate/endDate in YYYY-MM-DD format.',
                results: []
            };
        }

        try {
            const geoResponse = await axios.get('https://api.openweathermap.org/geo/1.0/direct', {
                params: {
                    q: country ? `${city},${country}` : city,
                    limit: 1,
                    appid: this.apiKey
                },
                timeout: 15000
            });

            if (!geoResponse.data?.length) {
                return { error: `City not found by OpenWeather: ${city}${country ? `, ${country}` : ''}`, results: [] };
            }

            const { lat, lon, name, country: resolvedCountry } = geoResponse.data[0];
            const response = await axios.get(`${this.baseUrl}/forecast`, {
                params: { lat, lon, units: 'metric', appid: this.apiKey },
                timeout: 15000
            });

            const daily = new Map();
            const requestedStartTime = new Date(`${requestedStart}T00:00:00Z`).getTime();
            const requestedEndTime = new Date(`${requestedEnd}T23:59:59Z`).getTime();

            // OpenWeather /forecast provides a rolling short-range forecast.
            // Choose the forecast entry closest to noon for each date rather than
            // requiring an exact "12:00" record, which can make valid forecasts
            // appear unavailable.
            for (const item of response.data?.list || []) {
                const timestamp = Number(item.dt) * 1000;
                if (!Number.isFinite(timestamp) || timestamp < requestedStartTime || timestamp > requestedEndTime) continue;

                const date = item.dt_txt?.split(' ')[0];
                if (!date) continue;

                const hour = Number(item.dt_txt?.split(' ')[1]?.slice(0, 2));
                const score = Math.abs((Number.isFinite(hour) ? hour : 0) - 12);
                const existing = daily.get(date);
                if (existing && existing._score <= score) continue;

                daily.set(date, {
                    _score: score,
                    date,
                    temperature: Math.round(Number(item.main?.temp)),
                    feelsLike: Math.round(Number(item.main?.feels_like)),
                    description: item.weather?.[0]?.description || null,
                    humidity: item.main?.humidity ?? null,
                    windSpeed: item.wind?.speed ?? null,
                    precipitationProbability: Number.isFinite(item.pop) ? Math.round(item.pop * 100) : null
                });
            }

            const results = Array.from(daily.values())
                .sort((a, b) => a.date.localeCompare(b.date))
                .map(({ _score, ...forecast }) => forecast);

            if (!results.length) {
                return {
                    error: `OpenWeather has no forecast coverage for ${requestedStart} to ${requestedEnd}. Its standard forecast endpoint only covers a short rolling window, so dates outside that window cannot be predicted live.`,
                    results: [],
                    location: { city: name, country: resolvedCountry, latitude: lat, longitude: lon }
                };
            }

            return {
                location: { city: name, country: resolvedCountry, latitude: lat, longitude: lon },
                coverage: {
                    requestedStart,
                    requestedEnd,
                    returnedStart: results[0].date,
                    returnedEnd: results[results.length - 1].date
                },
                results,
                source: 'OpenWeather 5-day / 3-hour forecast'
            };
        } catch (error) {
            const providerError = error.response?.data?.message || error.response?.data?.error || error.message;
            console.error('❌ Weather API Error:', error.response?.data || error.message);
            return {
                error: `Live weather request failed: ${providerError}`,
                results: []
            };
        }
    }
}
