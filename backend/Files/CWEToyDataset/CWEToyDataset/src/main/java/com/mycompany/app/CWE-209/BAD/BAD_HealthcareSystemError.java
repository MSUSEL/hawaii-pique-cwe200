// Define a custom exception
class NonCompliantSystemConfigurationException extends Exception {
    public NonCompliantSystemConfigurationException(String message) {
        super(message);
    }
}

public class BAD_HealthcareSystemError {
    public void initializeHealthcareSystem(String systemConfig) {
        try {
            if (!systemConfig.contains("HIPAA_compliant=true")) {
                String encryptionKey = "AESKey-Secret-9876543210";
                String privateCertPath = "/etc/ssl/private/healthcare_cert.pem";
                String serverIp = "192.168.1.100";
                String internalConfigFile = "/opt/healthcare/config/internal.conf";
                throw new NonCompliantSystemConfigurationException(
                        "System configuration must be HIPAA compliant. Current configuration: " + systemConfig +
                        ". Encryption Key: " + encryptionKey +
                        ", Private Certificate Path: " + privateCertPath +
                        ", Server IP: " + serverIp +
                        ", Internal Config File: " + internalConfigFile);
            }
            // Normal initialization logic
            System.out.println("Healthcare system initialized successfully with configuration: " + systemConfig);
        } catch (NonCompliantSystemConfigurationException e) {
            logError(e.getMessage());
        }
    }

    private void logError(String errorMessage) {
        // Log the error along with detailed sensitive data.
        System.err.println("Error: " + errorMessage);
    }

    public static void main(String[] args) {
        new BAD_HealthcareSystemError().initializeHealthcareSystem("encryption=weak, HIPAA_compliant=false");
    }
}
