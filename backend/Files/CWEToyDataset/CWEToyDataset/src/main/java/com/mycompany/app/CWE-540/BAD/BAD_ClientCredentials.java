public class BAD_ClientCredentials {
    // Client ID and Secret for OAuth hardcoded in source code
    public static final String CLIENT_ID = "client-id-1234";
    public static final String CLIENT_SECRET = "super-secret-client-key";

    public void authenticate() {
        System.out.println("Authenticating with client ID: " + CLIENT_ID + " and secret: " + CLIENT_SECRET);
    }
}
