import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;
import java.nio.charset.StandardCharsets;

public class GOOD_HealthInfoSubmissionGET {
    private static SecretKey secretKey;

    static {
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128); // Using AES with 128-bit keys.
            secretKey = keyGen.generateKey();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String encrypt(String data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encrypted = cipher.doFinal(data.getBytes());
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public static void bookMedicalAppointment(String patientId, String symptoms, String doctorId) {
        try {
            // Encrypt and URL encode the parameters to ensure they are properly formatted for a URL
            symptoms = URLEncoder.encode(encrypt(symptoms), StandardCharsets.UTF_8.toString());
            patientId = URLEncoder.encode(encrypt(patientId), StandardCharsets.UTF_8.toString());
            doctorId = URLEncoder.encode(encrypt(doctorId), StandardCharsets.UTF_8.toString());

            String urlString = "https://healthcare.example.com/bookAppointment?patientId=" + patientId +
                               "&symptoms=" + symptoms + "&doctorId=" + doctorId; // Using HTTPS and encrypted data
            
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
        bookMedicalAppointment("12345", "Fever, cough, and shortness of breath", "DrJohnDoe"); // Now with encrypted parameters
    }
}
