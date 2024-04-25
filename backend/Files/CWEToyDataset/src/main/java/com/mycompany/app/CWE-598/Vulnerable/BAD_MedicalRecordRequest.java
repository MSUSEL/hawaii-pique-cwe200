import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_MedicalRecordRequest {
    public static void requestMedicalRecord(String patientId) {
        try {
            String urlString = "http://example.com/medicalRecords?patientId=" + patientId; // Sensitive data in query
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
            System.out.println("Sending medical record request for patient ID: " + patientId);
            System.out.println("Response Code: " + connection.getResponseCode());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        requestMedicalRecord("987654321"); // Extremely risky!
    }
}
