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
            placesRadius = 10000
        } = args || {};

        const durationNights = this.calculateDuration(departDate, returnDate);
        const calendarDays = durationNights + 1;
        const travelers = Math.max(1, Number(passengers) || 1);

        const result = {
            request: {
                origin,
                destinationCity,
                destinationCountry,
                departDate,
                returnDate: returnDate || null,
                travelers,
                durationNights,
                calendarDays,
                budgetLevel
            },
            services: {}
        };

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
                    checkOut: returnDate,
                    adults: travelers
                })),
                this.safeCall('attractions', () => this.placesService.getAttractions(
                    destinationCity,
                    12,
                    placesRadius
                )),
                this.safeCall('restaurants', () => this.placesService.getRestaurants(
                    destinationCity,
                    12
                )),
                this.safeCall('weather', () => this.weatherService.getWeatherForecast({
                    city: destinationCity,
                    country: destinationCountry,
                    startDate: departDate,
                    endDate: returnDate
                })),
                this.safeCall('budget', () => this.calculateBudget({
                    destinations: [destinationCity],
                    duration: durationNights,
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

        result.liveDataSummary = this.calculateLiveSubtotal(flights, hotels, durationNights);

        return result;
    }

    calculateLiveSubtotal(flights, hotels, durationNights) {
        const flightOptions = Array.isArray(flights) ? flights : [];
        const hotelOptions = Array.isArray(hotels) ? hotels : [];

        const cheapestFlight = flightOptions
            .map(f => Number(f.price))
            .filter(Number.isFinite)
            .sort((a, b) => a - b)[0] ?? null;

        const cheapestHotelPerNight = hotelOptions
            .map(h => Number(h.price))
            .filter(Number.isFinite)
            .sort((a, b) => a - b)[0] ?? null;

        const hotelTotal = cheapestHotelPerNight !== null
            ? cheapestHotelPerNight * durationNights
            : null;

        const components = {};
        let subtotal = 0;
        let complete = true;

        if (cheapestFlight !== null) {
            components.cheapestFlight = cheapestFlight;
            subtotal += cheapestFlight;
        } else {
            complete = false;
        }

        if (hotelTotal !== null) {
            components.cheapestHotel = cheapestHotelPerNight;
            components.cheapestHotelTotal = hotelTotal;
            subtotal += hotelTotal;
        } else {
            complete = false;
        }

        return {
            currency: 'USD',
            cheapestLiveSubtotal: subtotal,
            components,
            complete,
            note: complete
                ? 'Subtotal uses the cheapest returned live flight offer plus the cheapest returned hotel nightly rate multiplied by the number of nights. It excludes food, local transport, activities, taxes/fees not included in provider prices, and other trip costs.'
                : 'Live subtotal is incomplete because a usable live flight or hotel price was not returned.'
        };
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
                reason: 'Currency conversion was not requested.'
            };
        }

        return this.safeCall('currency', () => this.currencyService.getExchangeRate({
            from,
            to,
            amount
        }));
    }

    calculateDuration(startDate, endDate) {
        if (!startDate || !endDate) return 0;
        const start = new Date(`${startDate}T00:00:00Z`);
        const end = new Date(`${endDate}T00:00:00Z`);
        const diff = Math.round((end - start) / 86400000);
        if (!Number.isFinite(diff) || diff < 1) {
            throw new Error('returnDate must be after departDate');
        }
        return diff;
    }
}
