import axios from 'axios';

export class CurrencyService {
    constructor() {
        this.apiKey = process.env.EXCHANGE_API_KEY || '';
        this.baseUrl = 'https://v6.exchangerate-api.com/v6';
    }

    async getExchangeRate(args) {
        const from = String(args?.from || '').trim().toUpperCase();
        const to = String(args?.to || '').trim().toUpperCase();
        const amount = Number(args?.amount ?? 1);

        if (!from || !to || from.length !== 3 || to.length !== 3) {
            return { error: 'Currency codes must be valid 3-letter ISO codes.', results: [] };
        }

        if (!Number.isFinite(amount) || amount < 0) {
            return { error: 'Amount must be a non-negative number.', results: [] };
        }

        if (!this.apiKey) {
            return {
                error: 'Live exchange rate unavailable: EXCHANGE_API_KEY is not configured.',
                results: []
            };
        }

        try {
            const url = `${this.baseUrl}/${this.apiKey}/pair/${from}/${to}/${amount}`;
            const response = await axios.get(url, { timeout: 15000 });

            if (response.data?.result !== 'success') {
                return {
                    error: response.data?.['error-type'] || 'Exchange rate provider returned an error.',
                    results: []
                };
            }

            return {
                from,
                to,
                amount,
                rate: response.data.conversion_rate,
                convertedAmount: response.data.conversion_result,
                lastUpdate: response.data.time_last_update_utc,
                source: 'ExchangeRate-API'
            };
        } catch (error) {
            console.error('❌ Currency API Error:', error.response?.data || error.message);
            return { error: `Live exchange rate request failed: ${error.message}`, results: [] };
        }
    }
}
