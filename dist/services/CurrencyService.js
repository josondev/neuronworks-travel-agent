import axios from 'axios';
export class CurrencyService {
    apiKey;
    baseUrl = 'https://v6.exchangerate-api.com/v6';
    constructor() {
        this.apiKey = process.env.EXCHANGE_API_KEY || '';
    }
    async getExchangeRate(args) {
        const { from, to, amount = 1 } = args;
        if (!this.apiKey) {
            return {
                from, to, amount,
                rate: 1.1,
                convertedAmount: amount * 1.1,
                note: "Mock data - API key missing"
            };
        }
        try {
            const url = `${this.baseUrl}/${this.apiKey}/pair/${from}/${to}/${amount}`;
            const response = await axios.get(url);
            return {
                from,
                to,
                amount,
                rate: response.data.conversion_rate,
                convertedAmount: response.data.conversion_result,
                lastUpdate: response.data.time_last_update_utc
            };
        }
        catch (error) {
            console.error("Currency API Error", error);
            return { error: "Failed to fetch exchange rate" };
        }
    }
}
//# sourceMappingURL=CurrencyService.js.map