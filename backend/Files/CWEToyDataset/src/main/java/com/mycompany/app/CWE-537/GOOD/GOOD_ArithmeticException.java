public class GOOD_ArithmeticException {
    public static void main(String[] args) {
        try {
            int sensitiveMultiplier = getSensitiveMultiplier();
            // Potential division by zero error
            int result = 100 / sensitiveMultiplier; 
        } catch (ArithmeticException e) {
            // Inappropriately handling ArithmeticException
            System.err.println("Failed to calculate due to an arithmetic error.");
            // Exposing sensitive business logic detail
            System.err.println("This occurred while using a sensitive business multiplier.");
        }
    }

    private static int getSensitiveMultiplier() {
        // Simulate retrieving a sensitive multiplier that is mistakenly zero
        return 0;  // Example of a critical mistake in business logic
    }
}
