import java.io.FileOutputStream;

public class GOOD_HealthRecordStorage {
    public static void storeHealthRecord(String patientName, String healthData) {
        String encryptedData = encryptHealthData(healthData);  // Assume encryption is handled securely
        try (FileOutputStream out = new FileOutputStream("/mnt/public_health_records/" + patientName + ".enc")) {
            out.write(encryptedData.getBytes());
        } catch (Exception e) {
            System.err.println("Failed to store health record for " + patientName);
        }
    }

    private static String encryptHealthData(String data) {
        // Encryption logic, simplified for example
        return "Encrypted(" + data + ")";
    }

    public static void main(String[] args) {
        storeHealthRecord("JohnDoe", "Sensitive Health Information");
    }
}
