import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public class BAD_HealthInfoSubmissionGET {
    
    public static void bookMedicalAppointment(String patientId, String symptoms, String doctorId) {
        try {
            String encodedSymptoms = URLEncoder.encode(symptoms, StandardCharsets.UTF_8.toString());

            String urlString = "http://healthcare.example.com/bookAppointment?patientId=" + patientId +
                               "&symptoms=" + encodedSymptoms + "&doctorId=" + doctorId;
            
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }
    
    public static void main(String[] args) {
        bookMedicalAppointment("12345", "Fever, cough, and shortness of breath", "DrJohnDoe"); // Highly sensitive and insecure!
    }
}
