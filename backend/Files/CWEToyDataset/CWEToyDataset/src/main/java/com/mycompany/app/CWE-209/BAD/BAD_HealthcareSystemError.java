public class BAD_HealthcareSystemError {
    public void initializeHealthcareSystem(String systemConfig) {
        try {
            if (!systemConfig.contains("HIPAA_compliant")) {
                throw new IllegalStateException("System configuration must be HIPAA compliant.");
            }
            // Initialization logic
        } catch (IllegalStateException e) {
            String errorMessage = "Initialization failed with non-compliant settings: " + systemConfig;
            System.err.println(errorMessage);
            throw new SecurityException(errorMessage);
        }
    }

    public static void main(String[] args) {
        new BAD_HealthcareSystemError().initializeHealthcareSystem("encryption=weak, HIPAA_compliant=false");
    }
}
