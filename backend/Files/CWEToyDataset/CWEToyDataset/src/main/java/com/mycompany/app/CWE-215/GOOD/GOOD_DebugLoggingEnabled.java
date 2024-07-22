public class GOOD_DebugLoggingEnabled {
    private static final boolean DEBUG_LOGGING_ENABLED = Boolean.parseBoolean(System.getenv("DEBUG_LOGGING_ENABLED"));

    public static void main(String[] args) {
        String accessToken = System.getenv("ACCESS_TOKEN");

        if (DEBUG_LOGGING_ENABLED) {
            System.out.println("DEBUG: Access Token.");
        }

    }
}
