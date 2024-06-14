public class BAD_ArithmeticExceptionHandling {
    public static void main(String[] args) {
        int sensitiveMultiplier = getSensitiveMultiplier();
        try {
            // Potential division by zero error
            int result = 100 / sensitiveMultiplier; 
        } catch (ArithmeticException e) {
            System.err.println("Failed to calculate due to an arithmetic error ussing " + String.valueOf(sensitiveMultiplier));
    }
}

    private static int getSensitiveMultiplier() {
        // Simulate retrieving a sensitive multiplier that is mistakenly zero
        return 0;  // Example of a critical mistake in business logic
    }
}
