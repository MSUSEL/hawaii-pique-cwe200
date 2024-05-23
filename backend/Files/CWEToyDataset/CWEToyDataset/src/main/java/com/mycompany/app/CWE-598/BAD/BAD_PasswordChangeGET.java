import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_PasswordChangeGET {
    
    public static void changePassword(String username, String oldPassword, String newPassword) {
        try {
            String urlString = "http://example.com/changePassword?username=" + username + 
                               "&oldPassword=" + oldPassword + "&newPassword=" + newPassword; // Sensitive data in query
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
        changePassword("user2", "oldPass123", "newPass456"); // Highly insecure method!
    }
}
