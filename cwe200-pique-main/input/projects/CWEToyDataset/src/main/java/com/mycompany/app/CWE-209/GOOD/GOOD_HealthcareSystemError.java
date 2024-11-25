public class GOOD_HealthcareSystemError {
    public void initializeHealthcareSystem(String systemConfig) {
        try {
            if (!systemConfig.contains("HIPAA_compliant")) {
                throw new IllegalStateException("Non-compliance with required standards detected.");
            }
            // Initialization logic
        } catch (IllegalStateException e) {
            System.err.println("Initialization failed due to compliance issues.");
            throw new SecurityException("System configuration issue.");
        }
    }

    public static void main(String[] args) {
        new GOOD_HealthcareSystemError().initializeHealthcareSystem("encryption=weak, HIPAA_compliant=false");
    }
}