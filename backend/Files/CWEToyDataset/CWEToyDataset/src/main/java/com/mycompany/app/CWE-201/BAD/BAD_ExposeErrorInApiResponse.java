import spark.Request;
import spark.Response;
import spark.Route;
import static spark.Spark.*;

public class BAD_ExposeErrorInApiResponse {
    public static void main(String[] args) {
        post("/api/process", new Route() {
            @Override
            public Object handle(Request request, Response response) throws Exception {
                try {
                    String data = request.queryParams("data");
                    processData(data);
                } catch (Exception e) {
                    // BAD: Exposing error details in API response
                    response.status(500);
                    return "Error processing request: " + e.getMessage();
                }
                return "Success";
            }
        });
    }

    private static void processData(String data) throws Exception {
        if (data == null) {
            throw new IllegalArgumentException("Data cannot be null");
        }
        // Data processing logic...
    }
}
