import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public class GOOD_HealthInfoSubmissionPOST {
    private static SecretKey secretKey;

    static {
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128); // Using AES with 128-bit keys.
            secretKey = keyGen.generateKey();
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static String encrypt(String data) throws Exception {
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, secretKey);
        byte[] encrypted = cipher.doFinal(data.getBytes());
        return Base64.getEncoder().encodeToString(encrypted);
    }

    public static void bookMedicalAppointment(String patientId, String symptoms, String doctorId) {
        HttpURLConnection connection = null;
        try {
            // Encrypt and URL encode the parameters
            patientId = URLEncoder.encode(encrypt(patientId), StandardCharsets.UTF_8.toString());
            symptoms = URLEncoder.encode(encrypt(symptoms), StandardCharsets.UTF_8.toString());
            doctorId = URLEncoder.encode(encrypt(doctorId), StandardCharsets.UTF_8.toString());

            URL url = new URL("https://healthcare.example.com/bookAppointment");
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setDoOutput(true); // To send request content
            connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");

            // Construct the data to be sent in the POST body
            String postData = "patientId=" + patientId + "&symptoms=" + symptoms + "&doctorId=" + doctorId;
            
            try (OutputStream os = connection.getOutputStream()) {
                os.write(postData.getBytes());
                os.flush();
            }

        } catch (Exception e) {
            System.err.println("Error occurred while booking medical appointment.");
        } finally {
            if (connection != null) {
                connection.disconnect();
            }
        }
    }
    
    public static void main(String[] args) {
        bookMedicalAppointment("12345", "Fever, cough, and shortness of breath", "DrJohnDoe"); // Now with encrypted parameters
    }
}
