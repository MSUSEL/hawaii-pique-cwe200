import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import java.io.IOException;
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

public class GOOD_PersonalDetailsGET_Encrypted {
    private static SecretKey secretKey;

    static {
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128); // AES with 128-bit key for encryption
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

    public static void sendPersonalDetails(String fullName, String dateOfBirth) {
        try {
            fullName = URLEncoder.encode(encrypt(fullName), StandardCharsets.UTF_8.name());
            dateOfBirth = URLEncoder.encode(encrypt(dateOfBirth), StandardCharsets.UTF_8.name());
            String url = "https://example.com/register?fullName=" + fullName + "&dob=" + dateOfBirth; // Encrypted data in query
            
            CloseableHttpClient httpClient = HttpClients.createDefault();
            HttpGet httpGet = new HttpGet(url);

            HttpResponse response = httpClient.execute(httpGet);
            System.out.println("Sending 'GET' request to URL: " + url);
            System.out.println("Response Code: " + response.getStatusLine().getStatusCode());

            httpClient.close();
        } catch (ClientProtocolException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        sendPersonalDetails("John Doe", "1980-01-01"); // Now with encrypted parameters
    }
}
