public class GOOD_ApplicationSettings {
    public void callService() {
        String serviceEndpoint = System.getenv("SERVICE_ENDPOINT");
        String accessKey = System.getenv("ACCESS_KEY");
        System.out.println("Calling service at.");
    }
}
