export class TripPlannerService {
    constructor({ flightService, accommodationService, placesService, weatherService, currencyService, calculateBudget }) {
        this.flightService = flightService;
        this.accommodationService = accommodationService;
        this.placesService = placesService;
        this.weatherService = weatherService;
        this.currencyService = currencyService;
        this.calculateBudget = calculateBudget;
    }

    async buildTripData(args) {
        const {
            origin,
            destinationCity,
            destinationCountry,
            departDate,
            returnDate,
            passengers = 1,
            budgetLevel = 'budget',
            currencyFrom,
            currencyTo,
            currencyAmount = 1,
            placesRadius = 5000
        } = args || {};

        const duration = this.calculateDuration(departDate, returnDate);
        const travelers = Math.max(1, Number(passengers) || 1);

        const result = {
            request: {
                origin,
                destinationCity,
                destinationCountry,
                departDate,
                returnDate: returnDate || null,
                travelers,
                durationDays: duration,
                budgetLevel
            },
            services: {}
        };

        // Run independent live services concurrently. A failure in one service
        // is captured as data instead of preventing the other services from running.
        const [flights, hotels, attractions, restaurants, weather, budget, currency] =
            await Promise.all([
                this.safeCall('flights', () => this.flightService.searchFlights({
                    origin,
                    destination: args.destinationAirport || destinationCity,
                    departDate,
                    returnDate,
                    passengers: travelers
                })),
                this.safeCall('hotels', () => this.accommodationService.searchAccommodation({
                    city: destinationCity,
                    checkIn: departDate,
                    checkOut: returnDate || departDate,
                    adults: travelers
                })),
                this.safeCall('attractions', () => this.placesService.getAttractions(
                    destinationCity,
                    10
                )),
                this.safeCall('restaurants', () => this.placesService.getRestaurants(
                    destinationCity,
                    10
                )),
                this.safeCall('weather', () => this.weatherService.getWeatherForecast({
                    city: destinationCity,
                    country: destinationCountry,
                    startDate: departDate,
                    endDate: returnDate || departDate
                })),
                this.safeCall('budget', () => this.calculateBudget({
                    destinations: [destinationCity],
                    duration,
                    travelers,
                    budgetLevel
                })),
                this.getCurrencyIfRequested(currencyFrom, currencyTo, currencyAmount)
            ]);

        result.services.flights = flights;
        result.services.hotels = hotels;
        result.services.attractions = attractions;
        result.services.restaurants = restaurants;
        result.services.weather = weather;
        result.services.budget = budget;
        result.services.currency = currency;

        return result;
    }

    async safeCall(name, fn) {
        try {
            return await fn();
        } catch (error) {
            return {
                error: `${name} service failed: ${error?.message || String(error)}`,
                results: []
            };
        }
    }

    async getCurrencyIfRequested(from, to, amount) {
        if (!from || !to) {
            return {
                skipped: true,
                reason: 'Currency conversion was not requested. Provide currencyFrom and currencyTo when conversion is needed.'
            };
        }

        return this.safeCall('currency', () => this.currencyService.getExchangeRate({
            from,
            to,
            amount
        }));
    }

    calculateDuration(startDate, endDate) {
        if (!startDate || !endDate) return 1;
        const start = new Date(`${startDate}T00:00:00Z`);
        const end = new Date(`${endDate}T00:00:00Z`);
        const diff = Math.ceil((end - start) / 86400000);
        if (!Number.isFinite(diff) || diff < 1) {
            throw new Error('returnDate must be after departDate');
        }
        return diff;
    }
}
