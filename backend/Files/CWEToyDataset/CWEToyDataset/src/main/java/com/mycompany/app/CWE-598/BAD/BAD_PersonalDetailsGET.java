import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.HttpResponse;
import java.io.IOException;

public class BAD_PersonalDetailsGET {
    public static void sendPersonalDetails(String fullName, String dateOfBirth) {
        try {
            String url = "http://example.com/register?fullName=" + fullName + "&dob=" + dateOfBirth; // Sensitive data in query
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
        }
    }

    public static void main(String[] args) {
        sendPersonalDetails("John Doe", "1980-01-01"); // Extremely risky!
    }
}
