import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class BAD_UserProfileManagerCustomSink {

    public static void updateUserProfile(String userId, String profileData) {
        String command = "update_user_profile --user-id " + userId + " --data " + profileData;

        try {
            Process process = Runtime.getRuntime().exec(command);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String error;

            while ((error = errorReader.readLine()) != null) {
                ShowMessage.show("User profile update error for user " + userId + ": " + error);
            }

            if (process.waitFor() != 0) {
                ShowMessage.show("User profile update failed for user " + userId + ", see logs for details.");
            }
        } catch (IOException | InterruptedException e) {
            ShowMessage.show("Update operation failed for user " + userId + ": " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        updateUserProfile("user123", "{ \"name\": \"John Doe\", \"email\": \"john.doe@example.com\" }");
    }
}
