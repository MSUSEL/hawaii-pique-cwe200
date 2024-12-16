public class BAD_SecurityException {
    public void handleRequest(String requestData) {
        try {
            if (requestData == null) {
                throw new SecurityException();
            } else {
                // Process the request
                this.processRequest(requestData);
            }
        } catch (SecurityException e) {
            System.err.println("Your request " + requestData + " is not authorized. Please provide valid request data.");
        }
    }

    private void processRequest(String requestData) {
        // Process the request
    }
}
