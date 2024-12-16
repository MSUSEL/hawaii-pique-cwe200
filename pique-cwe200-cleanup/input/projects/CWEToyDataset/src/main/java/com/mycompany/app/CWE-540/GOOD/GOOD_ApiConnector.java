public class GOOD_ApiConnector {
    private String apiUrl;
    private String apiKey;

    public GOOD_ApiConnector() {
        this.apiUrl = System.getenv("API_URL");
        this.apiKey = System.getenv("API_KEY");
    }

    public void connect() {
        System.out.println("Connecting to API at: " + apiUrl + " with API key: " + apiKey);
    }
}
