import java.net.HttpURLConnection;
import java.net.URL;

public class BAD_PasswordChangeGET {
    
    public static void changePassword(String username, String oldPassword, String newPassword) {
        try {
            String urlString = "http://secure.companyportal.com/changePassword?username=johndoe" + username + 
                               "&oldPassword=" + oldPassword + "&newPassword=" + newPassword;
            URL url = new URL(urlString);
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("GET");
            
        } catch (Exception e) {
            System.err.println("Error occurred");
        }
    }
    
    public static void main(String[] args) {
        changePassword("user2", "oldPass123", "newPass456");
    }
}
