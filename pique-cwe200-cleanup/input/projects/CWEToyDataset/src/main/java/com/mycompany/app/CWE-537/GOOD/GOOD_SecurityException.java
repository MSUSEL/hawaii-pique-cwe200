public class GOOD_SecurityException {
    public void handleRequest(String requestData) {
        try {
            if (requestData == null) {
                throw new SecurityException("Request data cannot be null.");
            }
            // Process the request
        } catch (SecurityException e) {
            throw new SecurityException("Error processing request.");
        }
    }
}
