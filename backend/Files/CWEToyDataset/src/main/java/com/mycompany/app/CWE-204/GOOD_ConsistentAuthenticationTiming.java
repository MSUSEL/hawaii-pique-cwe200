public class GOOD_ConsistentAuthenticationTiming {
    public static boolean consistentAuthenticate(String username, String password) {
        boolean validUser = username.equals("validUser");
        boolean correctPassword = password.equals("secretPassword");

        // Introducing a consistent delay for all authentication attempts
        try {
            Thread.sleep(1000); // Fixed delay to mask timing discrepancies
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        return validUser && correctPassword;
    }

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        consistentAuthenticate("validUser", "wrongPassword");
        long endTime = System.currentTimeMillis();
    }
}
