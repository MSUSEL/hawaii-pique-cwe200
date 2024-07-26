public class BAD_NetworkDebugger {
    private static final boolean NETWORK_DEBUG_MODE = System.getProperty("NETWORK_DEBUG_MODE", "false").equalsIgnoreCase("true");

    public static void main(String[] args) {
        if (NETWORK_DEBUG_MODE) {
            testConnection("192.168.1.100", "b1234567-89ab-cdef-0123-456789abcdef");
        }
    }

    private static void testConnection(String ipAddress, String apiKey) {
        System.out.println("DEBUG: Testing connection to IP: " + ipAddress + " using API Key: " + apiKey);
        // Assume network testing logic here
    }
}
