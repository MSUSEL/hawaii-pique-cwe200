import java.io.BufferedReader;
import java.io.InputStreamReader;

public class BAD_SensitiveInfoExposureViaShellError {

    public static void main(String[] args) {
        try {
            String password = "GoldenSunrise2023";
            String[] command = {"/bin/sh", "-c", "somecommand --password=", password};
            Process process = Runtime.getRuntime().exec(command);
            process.waitFor();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String line = "";
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
