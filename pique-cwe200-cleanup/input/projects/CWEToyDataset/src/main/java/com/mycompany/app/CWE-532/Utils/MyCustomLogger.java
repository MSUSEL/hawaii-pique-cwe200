public class MyCustomLogger {

    public static void logInfo(String message) {
        System.out.println("INFO: " + message);
    }

    public static void logError(String message) {
        System.out.println("ERROR: " + message);
    }

    public static void logDebug(String message) {
        System.out.println("DEBUG: " + message);
    }
}
