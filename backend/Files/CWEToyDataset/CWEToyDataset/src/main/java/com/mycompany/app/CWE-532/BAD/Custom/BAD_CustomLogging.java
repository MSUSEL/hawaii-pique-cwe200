import MyCustomLogger;

public class BAD_CustomLogging {
   
    public void logAPIUsage(String apiKey, String methodName) {
        MyCustomLogger logger = MyCustomLogger.getInstance();
        logger("API usage: Key: " + apiKey + ", Method: " + methodName);
    }
    
}
