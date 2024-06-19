import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_MedicalRecordRequest {
    public static void requestMedicalRecord(String patientId) {
        try {
            String urlString = "http://example.com/medicalRecords?patientId=" + patientId;
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        requestMedicalRecord("987654321");
    }
}
