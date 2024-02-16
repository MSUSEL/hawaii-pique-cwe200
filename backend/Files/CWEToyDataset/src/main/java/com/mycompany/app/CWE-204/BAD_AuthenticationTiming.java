public class BAD_AuthenticationTiming {
    public static boolean authenticate(String username, String password) {
        // Simulate database lookup that takes longer for valid users
        if (username.equals("validUser")) {
            try {
                Thread.sleep(2000); // Artificial delay for demonstration
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            return password.equals("secretPassword");
        }
        return false;
    }

    public static void main(String[] args) {
        long startTime = System.currentTimeMillis();
        authenticate("validUser", "wrongPassword");
        long endTime = System.currentTimeMillis();
        System.out.println("Authentication took: " + (endTime - startTime) + "ms");
    }
}
