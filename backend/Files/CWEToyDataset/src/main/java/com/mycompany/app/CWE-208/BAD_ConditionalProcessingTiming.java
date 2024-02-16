public class BAD_ConditionalProcessingTiming {
    // Vulnerability: The time taken to process requests varies depending on whether a secret value matches a part of the request.
    // This discrepancy can be observed and exploited by an attacker to infer the secret value through timing analysis.

    private static final String SECRET_TOKEN = "SecretToken123";

    public static boolean verifyToken(String userToken) {
        // Direct comparison introduces a timing discrepancy
        for (int i = 0; i < userToken.length(); i++) {
            if (i >= SECRET_TOKEN.length() || userToken.charAt(i) != SECRET_TOKEN.charAt(i)) {
                // Processing stops as soon as a mismatch is found, which can vary the time taken
                return false;
            }
            // Simulate additional processing time for each matching character
            try {
                Thread.sleep(10); // Artificial delay to exaggerate the timing difference
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        return userToken.equals(SECRET_TOKEN);
    }

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        verifyToken("Secret");
        long endTime = System.currentTimeMillis();
        System.out.println("Token verification took: " + (endTime - startTime) + "ms");
    }
}
