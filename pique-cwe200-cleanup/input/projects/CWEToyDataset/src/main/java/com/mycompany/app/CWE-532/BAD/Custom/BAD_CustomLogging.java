public class BAD_CustomLogging {
   
    public void logAPIUsage(String apiKey, String methodName) {
        MyCustomLogger.logInfo("API usage: Key: " + apiKey + ", Method: " + methodName);
    }
    
}
