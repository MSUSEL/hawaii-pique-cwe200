public class BAD_SecurityException {
    public void handleRequest(String requestData) {
            if (requestData == null) {
                throw new SecurityException("Your request " + requestData + " is not authorized. Please provide valid request data.");
            }
            // Process the request
            else{
                this.processRequest(requestData);
            }
    }

    private void processRequest(String requestData) {
        // Process the request
    }
}
