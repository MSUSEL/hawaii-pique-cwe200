public class TraceKeeper {
    public static void enterMethod(String methodName) {
        System.out.println("Entering Method: " + methodName);
    }

    public static void exitMethod(String methodName) {
        System.out.println("Exiting Method: " + methodName);
    }
}
