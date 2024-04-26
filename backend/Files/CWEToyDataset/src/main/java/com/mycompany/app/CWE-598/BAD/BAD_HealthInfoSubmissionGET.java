import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public class BAD_HealthInfoSubmissionGET {
    
    public static void bookMedicalAppointment(String patientId, String symptoms, String doctorId) {
        try {
            // URL encode the parameters to ensure they are properly formatted for a URL
            String encodedSymptoms = URLEncoder.encode(symptoms, StandardCharsets.UTF_8.toString());

            String urlString = "http://healthcare.example.com/bookAppointment?patientId=" + patientId +
                               "&symptoms=" + encodedSymptoms + "&doctorId=" + doctorId; // Exposing PHI in query
            
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
            System.out.println("Sending 'GET' request to URL : " + url);
            System.out.println("Response Code : " + connection.getResponseCode());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static void main(String[] args) {
        bookMedicalAppointment("12345", "Fever, cough, and shortness of breath", "DrJohnDoe"); // Highly sensitive and insecure!
    }
}
