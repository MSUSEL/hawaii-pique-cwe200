public class BAD_SecurityException {
    public void handleRequest(String requestData) {
        try {
            if (requestData == null) {
                throw new SecurityException("Request data cannot be null.");
            }
            // Process the request
        } catch (SecurityException e) {
            System.err.println("Failed to process request due to missing data." + e.getMessage() + ". Data received: " + requestData);
            throw new SecurityException("Error processing request: " + e.getMessage() + ". Data received: " + requestData);
        }
    }
}
