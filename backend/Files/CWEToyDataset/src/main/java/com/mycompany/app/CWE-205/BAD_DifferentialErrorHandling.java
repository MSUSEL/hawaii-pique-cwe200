public class BAD_DifferentialErrorHandling {
    // Vulnerability: Observable discrepancy in error handling reveals information about internal processing.
    // Errors thrown for invalid data might contain stack traces or messages indicating the nature of the data or its processing path.

    public static void processData(String data) {
        try {
            validateData(data); // Throws different exceptions based on data validity
        } catch (InvalidDataException e) {
            // Exposing detailed exception messages can reveal insights into the validation logic or existence of certain data.
            System.err.println("Validation error: " + e.getMessage()); // Specific validation error message
        } catch (Exception e) {
            System.err.println("General processing error."); // Generic error message for other exceptions
        }
    }

    private static void validateData(String data) throws InvalidDataException {
        // Placeholder for data validation logic that throws exceptions based on specific validation failures
        if (data.isEmpty()) {
            throw new InvalidDataException("Data cannot be empty.");
        }
        // Additional validation checks...
    }

    static class InvalidDataException extends Exception {
        public InvalidDataException(String message) {
            super(message);
        }
    }

    public static void main(String[] args) {
        processData(""); // Reveals that data cannot be empty based on the specific error message.
    }
}
