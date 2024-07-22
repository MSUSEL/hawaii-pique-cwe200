public class GOOD_APIEndpointDebugging {
    private static final boolean API_DEBUG = Boolean.parseBoolean(System.getenv("API_DEBUG"));

    public static void main(String[] args) {
        String apiEndpoint = "https://api.example.com/data";
        String apiKey = "api_key_123456789";

        if (API_DEBUG) {
            testAPI(apiEndpoint, apiKey);
        }
    }

    private static void testAPI(String url, String key) {
        System.out.println("DEBUG: Testing API endpoint.");
        // Assume some testing logic here
    }
}
