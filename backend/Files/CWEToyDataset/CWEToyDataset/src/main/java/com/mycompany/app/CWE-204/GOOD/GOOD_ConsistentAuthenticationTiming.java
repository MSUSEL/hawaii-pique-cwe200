public class GOOD_ConsistentAuthenticationTiming {
    
    private static final String VALID_USERNAME = System.getenv("VALID_USERNAME");
    private static final String VALID_PASSWORD = System.getenv("VALID_PASSWORD");

    public static boolean consistentAuthenticate(String attemptUsername, String attemptPassword) {
        boolean validUser = attemptUsername.equals(VALID_USERNAME);
        boolean correctPassword = attemptPassword.equals(VALID_PASSWORD);

        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return validUser && correctPassword;
    }

    public static void main(String[] args) {
        consistentAuthenticate(System.getenv("ATTEMPT_USERNAME"), System.getenv("ATTEMPT_PASSWORD"));
    }
}
