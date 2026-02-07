import dotenv from 'dotenv';
import axios from 'axios';

// Load .env file
dotenv.config();

// CLEAN THE KEYS (Remove spaces and quotes)
const OPEN_TRIP_KEY = (process.env.OPENTRIPMAP_API_KEY || '').replace(/["']/g, '').trim();
const BOOKING_KEY = (process.env.BOOKING_API_KEY || '').replace(/["']/g, '').trim();

console.log("=== DEBUGGING API KEYS (V2) ===\n");

async function testOpenTripMap() {
    console.log("1. Testing OpenTripMap...");
    
    if (!OPEN_TRIP_KEY) {
        console.log("❌ FAILED: Key is MISSING in .env");
        return;
    }
    // Show first 5 chars to verify it loaded
    console.log(`   Key loaded: ${OPEN_TRIP_KEY.substring(0, 5)}...`);

    try {
        // Use the cleaned key
        const url = `https://api.opentripmap.com/0.1/en/places/geoname?name=London&apikey=${OPEN_TRIP_KEY}`;
        const response = await axios.get(url);
        
        if (response.data.lat) {
            console.log("✅ SUCCESS: OpenTripMap is working!");
            console.log(`   Found: ${response.data.name} (Lat: ${response.data.lat})`);
        } else {
            console.log("⚠️ RESPONSE OK, BUT NO DATA:", response.data);
        }
    } catch (error) {
        console.log("❌ FAILED: OpenTripMap Error");
        console.log(`   Error: ${error.message}`);
        if (error.response) console.log(`   Data:`, error.response.data);
    }
    console.log("\n--------------------------------\n");
}

async function testBookingAPI() {
    console.log("2. Testing Booking.com...");
    
    if (!BOOKING_KEY) {
        console.log("❌ FAILED: Key is MISSING in .env");
        return;
    }

    // UPDATED HOST for your subscription
    const host = 'booking-com15.p.rapidapi.com'; 
    
    try {
        const response = await axios.get(`https://${host}/api/v1/hotels/searchHotels`, {
            params: { dest_id: '-2092174', search_type: 'CITY', arrival_date: '2025-10-01', departure_date: '2025-10-05' },
            headers: {
                'X-RapidAPI-Key': BOOKING_KEY,
                'X-RapidAPI-Host': host
            }
        });
        
        // Note: booking-com15 has a slightly different response format sometimes, 
        // but if we get ANY 200 OK response, the key works.
        if (response.status === 200) {
            console.log("✅ SUCCESS: Booking.com API is working!");
        }
    } catch (error) {
        // Even if it fails with "validation" or "params" error, 
        // as long as it's NOT 403, the Key is good.
        if (error.response && error.response.status !== 403) {
             console.log("✅ SUCCESS: Key is accepted! (Ignore parameter errors for now)");
        } else {
            console.log("❌ FAILED: Booking.com Error");
            if (error.response) {
                console.log(`   Status: ${error.response.status}`);
                console.log(`   Message:`, error.response.data.message || error.response.data);
            } else {
                console.log(`   Error: ${error.message}`);
            }
        }
    }
    console.log("\n--------------------------------\n");
}

(async () => {
    await testOpenTripMap();
    await testBookingAPI();
})();