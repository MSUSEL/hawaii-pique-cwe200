import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
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

public class GOOD_PersonalDetailsPOST_Encrypted {
    private static SecretKey secretKey;

    static {
        try {
            KeyGenerator keyGen = KeyGenerator.getInstance("AES");
            keyGen.init(128); // AES with 128-bit key for encryption
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

    public static void sendPersonalDetails(String fullName, String dateOfBirth) {
        try {
            fullName = URLEncoder.encode(encrypt(fullName), StandardCharsets.UTF_8.name());
            dateOfBirth = URLEncoder.encode(encrypt(dateOfBirth), StandardCharsets.UTF_8.name());

            String url = "https://example.com/register";
            
            CloseableHttpClient httpClient = HttpClients.createDefault();
            HttpPost httpPost = new HttpPost(url);
            httpPost.setHeader("Content-Type", "application/x-www-form-urlencoded");
            String postData = "fullName=" + fullName + "&dob=" + dateOfBirth;

            httpPost.setEntity(new StringEntity(postData));

            HttpResponse response = httpClient.execute(httpPost);
            httpClient.close();
        } catch (ClientProtocolException e) {
            System.err.println("Error occurred");
        } catch (IOException e) {
            System.err.println("Error occurred");
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        sendPersonalDetails("John Doe", "1980-01-01");
    }
}
