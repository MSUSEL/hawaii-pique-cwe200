public class GOOD_FixedResponseTiming {
    // Mitigation: Implementing fixed response timing to mask timing discrepancies that could indicate system behavior.
    // This practice makes it harder for attackers to use timing analysis to infer system states or validate guesses.

    public static boolean login(String username, String password) {
        long start = System.nanoTime();
        boolean result = "admin".equals(username) && "password".equals(password);
        // Ensure the method takes a consistent amount of time to return, regardless of the outcome.
        try {
            long elapsed = System.nanoTime() - start;
            long delay = 1000000L - elapsed; // Adjust this value as needed to normalize the timing.
            if (delay > 0) {
                Thread.sleep(delay / 1000000L, (int) (delay % 1000000L));
            }
        } catch (InterruptedException e) {
            System.err.println("An error has occurred.");
        }
        return result;
    }

    public static void main(String[] args) {
        System.out.println(login("admin", "wrongpassword")); // Timing analysis will not reveal if the username is correct.
    }
}
