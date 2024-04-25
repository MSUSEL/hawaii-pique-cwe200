public class GOOD_DebugFeatureToggle {
    private static final boolean DEBUG_FEATURE_TOGGLE = Boolean.parseBoolean(System.getProperty("DEBUG_FEATURE_TOGGLE", "false"));

    public static void main(String[] args) {
        String userSSN = "123-45-6789"; // Sensitive personal information

        if (DEBUG_FEATURE_TOGGLE) {
            System.out.println("DEBUG: User SSN.");
        }

        // Code to handle sensitive personal information
    }
}
