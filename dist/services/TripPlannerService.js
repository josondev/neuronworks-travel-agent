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
            destinationAirport,
            departDate,
            returnDate,
            passengers = 1,
            budgetLevel = 'budget',
            currencyFrom,
            currencyTo,
            currencyAmount = 1,
            placesRadius = 10000
        } = args || {};

        const dateCheck = this.validateDates(departDate, returnDate);
        const travelers = Math.max(1, Number(passengers) || 1);

        if (!dateCheck.ok) {
            return {
                planningBlocked: true,
                error: dateCheck.error,
                request: {
                    origin,
                    destinationAirport,
                    destinationCity,
                    destinationCountry,
                    departDate,
                    returnDate: returnDate || null,
                    travelers,
                    durationNights: null,
                    calendarDays: null,
                    budgetLevel
                },
                services: {
                    flights: { error: dateCheck.error, results: [] },
                    hotels: { error: dateCheck.error, results: [] },
                    weather: { error: dateCheck.error, results: [] }
                }
            };
        }

        const durationNights = this.calculateDuration(departDate, returnDate);
        const calendarDays = durationNights + 1;

        const result = {
            request: {
                origin,
                destinationAirport,
                destinationCity,
                destinationCountry,
                departDate,
                returnDate,
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
                    destination: destinationAirport || destinationCity,
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
                    8,
                    placesRadius
                )),
                this.safeCall('restaurants', () => this.placesService.getRestaurants(
                    destinationCity,
                    8
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

    validateDates(departDate, returnDate) {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(String(departDate || '')) || !/^\d{4}-\d{2}-\d{2}$/.test(String(returnDate || ''))) {
            return { ok: false, error: 'Please provide departure and return dates in YYYY-MM-DD format.' };
        }

        const start = new Date(`${departDate}T00:00:00Z`);
        const end = new Date(`${returnDate}T00:00:00Z`);
        const now = new Date();
        const today = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));

        if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
            return { ok: false, error: 'Invalid travel dates.' };
        }
        if (start < today) {
            return { ok: false, error: `The departure date ${departDate} is in the past. Today is ${today.toISOString().slice(0, 10)}. Please choose a future date.` };
        }
        if (end < today) {
            return { ok: false, error: `The return date ${returnDate} is in the past. Today is ${today.toISOString().slice(0, 10)}. Please choose a future date.` };
        }
        if (end <= start) {
            return { ok: false, error: 'Return date must be after departure date.' };
        }
        return { ok: true };
    }

    calculateLiveSubtotal(flights, hotels, durationNights) {
        const flightOptions = Array.isArray(flights) ? flights : [];
        const hotelOptions = Array.isArray(hotels) ? hotels : [];
        const cheapestFlight = flightOptions.map(f => Number(f.price)).filter(Number.isFinite).sort((a, b) => a - b)[0] ?? null;
        const cheapestHotelPerNight = hotelOptions.map(h => Number(h.price)).filter(Number.isFinite).sort((a, b) => a - b)[0] ?? null;
        const hotelTotal = cheapestHotelPerNight === null ? null : cheapestHotelPerNight * durationNights;
        const components = {};
        let subtotal = 0;
        let complete = true;
        if (cheapestFlight !== null) { components.cheapestFlight = cheapestFlight; subtotal += cheapestFlight; } else complete = false;
        if (hotelTotal !== null) { components.cheapestHotelPerNight = cheapestHotelPerNight; components.cheapestHotelTotal = hotelTotal; subtotal += hotelTotal; } else complete = false;
        return {
            currency: 'USD',
            cheapestLiveSubtotal: subtotal,
            components,
            complete,
            note: complete
                ? 'Cheapest returned live flight plus cheapest returned hotel rate multiplied by nights. Excludes food, transport, activities, taxes/fees not included by providers, and other trip costs.'
                : 'Live subtotal is incomplete because a usable live flight or hotel price was not returned.'
        };
    }

    async safeCall(name, fn) {
        try { return await fn(); }
        catch (error) { return { error: `${name} service failed: ${error?.message || String(error)}`, results: [] }; }
    }

    async getCurrencyIfRequested(from, to, amount) {
        if (!from || !to) return { skipped: true, reason: 'Currency conversion was not requested.' };
        return this.safeCall('currency', () => this.currencyService.getExchangeRate({ from, to, amount }));
    }

    calculateDuration(startDate, endDate) {
        const start = new Date(`${startDate}T00:00:00Z`);
        const end = new Date(`${endDate}T00:00:00Z`);
        const diff = Math.round((end - start) / 86400000);
        if (!Number.isFinite(diff) || diff < 1) throw new Error('returnDate must be after departDate');
        return diff;
    }
}
