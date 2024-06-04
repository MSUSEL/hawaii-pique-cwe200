import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class DatabaseConnectorCustomSink {

    public static void connectToDatabase(String dbConnectionString) {
        String command = "connect_db --connection-string " + dbConnectionString;

        try {
            Process process = Runtime.getRuntime().exec(command);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String error;

            while ((error = errorReader.readLine()) != null) {
                PrintToConsole.print("Database connection error: " + error);
            }

            if (process.waitFor() != 0) {
                PrintToConsole.print("Database connection failed, see logs for details.");
            }
        } catch (IOException | InterruptedException e) {
            PrintToConsole.print("Connection operation failed: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        connectToDatabase("jdbc:mysql://localhost:3306/mydb?user=admin&password=secret");
    }
}
