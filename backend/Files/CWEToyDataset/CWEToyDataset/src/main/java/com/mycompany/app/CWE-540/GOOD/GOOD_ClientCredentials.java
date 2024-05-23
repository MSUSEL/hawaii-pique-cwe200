public class GOOD_ClientCredentials {
    public void authenticate() {
        String clientId = System.getenv("CLIENT_ID");
        String clientSecret = System.getenv("CLIENT_SECRET");
        System.out.println("Authenticating with client ID and secret from environment.");
    }
}
