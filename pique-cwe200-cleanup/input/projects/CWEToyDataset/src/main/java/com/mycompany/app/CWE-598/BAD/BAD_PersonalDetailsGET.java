import org.apache.http.client.methods.HttpGet;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.HttpResponse;
import java.io.IOException;

public class BAD_PersonalDetailsGET {
    public static void sendPersonalDetails(String fullName, String dateOfBirth) {
        try {
            String url = "http://userportal.acmeinc.com/register?fullName=" + fullName + "&dob=" + dateOfBirth;
            CloseableHttpClient httpClient = HttpClients.createDefault();
            HttpGet httpGet = new HttpGet(url);

            HttpResponse response = httpClient.execute(httpGet);

            httpClient.close();
        } catch (ClientProtocolException e) {
            System.err.println("Error occurred");
        } catch (IOException e) {
            System.err.println("Error occurred");
        }
    }

    public static void main(String[] args) {
        sendPersonalDetails("John Doe", "1980-01-01");
    }
}
