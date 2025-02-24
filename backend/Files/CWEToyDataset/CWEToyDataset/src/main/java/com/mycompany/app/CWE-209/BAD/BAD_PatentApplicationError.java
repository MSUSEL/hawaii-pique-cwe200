public class BAD_PatentApplicationError {
    public void submitPatentApplication(String patentDetails) {
        try {
            if (patentDetails == null) {
                // Simulated sensitive internal details that are concatenated into the error message.
                String applicationID = "PatentAppID-2025-7890";
                String internalFilePath = "/var/app/patents/submissions/patent_7890.conf";
                String inventorSSN = "SSN: 123-45-6789";
                String internalConfig = "Config: UseEncryption=true; EncryptionKey=SolarPanelKey-Secret!";
                throw new IllegalArgumentException("Patent details are required. " +
                        "Application ID: " + applicationID +
                        ", Internal File: " + internalFilePath +
                        ", Inventor Details: " + inventorSSN +
                        ", Internal Config: " + internalConfig);
            }
            // Simulated submission logic would normally process the patent application.
            System.out.println("Patent application submitted successfully: " + patentDetails);
        } catch (Exception e) {
            // Exposing sensitive error details in the log.
            System.err.println("Failed to submit patent application due to: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        // Example patent details that could be sensitive
        new BAD_PatentApplicationError().submitPatentApplication("Invention: New efficient solar panel, Inventor: John Doe");
    }
}
