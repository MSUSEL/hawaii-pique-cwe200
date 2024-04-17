import java.io.FileWriter;
import java.io.IOException;

public class BAD_WriteSensitiveUserInfo {
    private static final String LOG_FILE_PATH = "logs/sensitive_info.log";

    public static void main(String[] args) {
        String mobileNumber = "123-456-7890"; 
        String emailAddress = "user@example.com"; 
        String homeAddress = "123 Main St, Honolulu, USA";

        try {
            logSensitiveInfo(mobileNumber, emailAddress, homeAddress);
        } catch (IOException e) {
            System.err.println("Failed to log sensitive information: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void logSensitiveInfo(String mobile, String email, String address) throws IOException {
        try (FileWriter writer = new FileWriter(LOG_FILE_PATH, true)) {
            writer.write("User Details - Mobile: " + mobile + ", Email: " + email + ", Address: " + address + "\n");
        }
    }
}
