import java.util.logging.*;

public class BAD_APIAccessLogger {
    private static final Logger logger = Logger.getLogger("APIAccessLogger");

    public void logAPIUsage(String apiKey, String methodName) {
        // Log API usage
        logger.warning("API usage: Key: " + apiKey + ", Method: " + methodName);
    }
}
