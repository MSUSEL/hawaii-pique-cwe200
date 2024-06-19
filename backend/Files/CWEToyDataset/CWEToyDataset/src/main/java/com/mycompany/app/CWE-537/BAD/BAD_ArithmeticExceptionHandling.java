public class BAD_ArithmeticExceptionHandling {
    public static void main(String[] args) {
        int sensitiveMultiplier = getSensitiveMultiplier();
        
        if (sensitiveMultiplier == 0) {
            throw new ArithmeticException("Cannot calculate due to a sensitive multiplier of zero" + String.valueOf(sensitiveMultiplier));
        }
        else{
            System.out.println("Calculating with a sensitive multiplier of " + String.valueOf(sensitiveMultiplier));
        }
    }

    private static int getSensitiveMultiplier() {
        // Simulate retrieving a sensitive multiplier that is mistakenly zero
        return 0;  // Example of a critical mistake in business logic
    }
}
