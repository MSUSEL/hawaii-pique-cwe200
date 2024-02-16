public class BAD_EmbeddedSensitiveDataInTests {
    // This class demonstrates an insecure practice where sensitive data, such as personal identifiable information (PII),
    // is hardcoded directly into automated test scripts.

    public static void main(String[] args) {
        // Directly using sensitive personal information within a test case
        String personalEmail = "user@example.com"; // Sensitive PII hardcoded
        String personalPhone = "555-0100"; // Sensitive PII hardcoded

        if(!validatePersonalInfo(personalEmail, personalPhone)) {
            System.out.println("Validation failed");
        } else {
            System.out.println("Validation passed");
        }
    }

    private static boolean validatePersonalInfo(String email, String phone) {
        // Placeholder for validation logic
        // In a real scenario, this might involve format checks or mock service interactions
        return email.contains("@") && phone.matches("\\d{3}-\\d{4}");
    }
}
