// Define a custom exception
class NonCompliantSystemConfigurationException extends Exception {
    public NonCompliantSystemConfigurationException(String message) {
        super(message);
    }
}

public class BAD_HealthcareSystemError {
    public void initializeHealthcareSystem(String systemConfig) {
        try {
            if (!systemConfig.contains("HIPAA_compliant")) {
                throw new NonCompliantSystemConfigurationException(
                        "System configuration must be HIPAA compliant. Current configuration: " + systemConfig);
            }
            // Initialization logic
            System.out.println("Healthcare system initialized successfully with configuration: " + systemConfig);
        } catch (NonCompliantSystemConfigurationException e) {
            logError(e.getMessage());
        }
    }

    private void logError(String errorMessage) {
        // Log the error to a secure logging system instead of printing it directly
        System.err.println("Error: " + errorMessage); // This could be replaced with a logging framework like Log4j
    }

    public static void main(String[] args) {
        new BAD_HealthcareSystemError().initializeHealthcareSystem("encryption=weak, HIPAA_compliant=false");
    }
}