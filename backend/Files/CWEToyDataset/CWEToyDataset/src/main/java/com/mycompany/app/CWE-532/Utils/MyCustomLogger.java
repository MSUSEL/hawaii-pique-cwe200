public class MyCustomLogger {
    private static final MyCustomLogger instance = new MyCustomLogger();

    private MyCustomLogger() {
        // Private constructor to prevent instantiation
    }

    public static MyCustomLogger getInstance() {
        return instance;
    }

    public void logInfo(String message) {
        System.out.println("INFO: " + message);
    }

    public void logError(String message) {
        System.out.println("ERROR: " + message);
    }

    public void logDebug(String message) {
        System.out.println("DEBUG: " + message);
    }
}
