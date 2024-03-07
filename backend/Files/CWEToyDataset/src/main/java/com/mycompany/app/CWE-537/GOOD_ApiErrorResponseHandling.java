import spark.Service;

public class GOOD_ApiErrorResponseHandling {

    public static void main(String[] args) {
        Service http = Service.ignite().port(8080);

        http.get("/secure-data", (request, response) -> {
            try {
                // Attempt to perform an operation that could fail
                return secureDataFetch();
            } catch (Exception e) {
                // Secure: Only log the specific error and return a generic error message to the client
                System.err.println("Error fetching secure data: ");
                response.status(500);
                return "Unable to process the request due to a server error.";
            }
        });
    }

    private static String secureDataFetch() {
        // Simulated secure data fetching logic
        return "Sensitive data";
    }
}
