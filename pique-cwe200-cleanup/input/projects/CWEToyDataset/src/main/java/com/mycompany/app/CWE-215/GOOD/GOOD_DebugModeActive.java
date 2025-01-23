public class GOOD_DebugModeActive {
    private static final boolean DEBUG_MODE_ACTIVE = System.getProperty("DEBUG_MODE_ACTIVE", "false").equalsIgnoreCase("true");

    public static void main(String[] args) {
        String encryptionKey = "ABCD-1234-EFGH-5678";

        // Debug condition
        if (DEBUG_MODE_ACTIVE) {
            System.out.println("DEBUG: Using encryption key.");
        }

        // Placeholder for encryption routine
    }
}
