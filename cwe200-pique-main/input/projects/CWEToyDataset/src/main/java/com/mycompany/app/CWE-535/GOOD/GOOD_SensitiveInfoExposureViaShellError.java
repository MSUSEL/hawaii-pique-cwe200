import java.io.BufferedReader;
import java.io.InputStreamReader;

public class GOOD_SensitiveInfoExposureViaShellError {

    public static void main(String[] args) {
        try {
            String[] command = {"/bin/sh", "-c", "somecommand --password=SunnyVale2025"};
            Process process = Runtime.getRuntime().exec(command);
            process.waitFor();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
          System.err.println("An error occurred.");
        } catch (Exception e) {
            System.err.println("An error occurred.");
        }
    }
}
