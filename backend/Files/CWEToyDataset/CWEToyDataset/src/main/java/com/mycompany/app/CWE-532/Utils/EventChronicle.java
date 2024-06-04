import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class EventChronicle {

    private static final String LOG_FILE = "event_chronicle.log";
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public static void logEvent(String event) {
        try (FileWriter fileWriter = new FileWriter(LOG_FILE, true);
             PrintWriter printWriter = new PrintWriter(fileWriter)) {
            
            String timestamp = LocalDateTime.now().format(DATE_FORMATTER);
            printWriter.println(timestamp + " - " + event);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void logError(String error) {
        try (FileWriter fileWriter = new FileWriter(LOG_FILE, true);
             PrintWriter printWriter = new PrintWriter(fileWriter)) {
            
            String timestamp = LocalDateTime.now().format(DATE_FORMATTER);
            printWriter.println(timestamp + " - ERROR: " + error);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void logInfo(String info) {
        try (FileWriter fileWriter = new FileWriter(LOG_FILE, true);
             PrintWriter printWriter = new PrintWriter(fileWriter)) {
            
            String timestamp = LocalDateTime.now().format(DATE_FORMATTER);
            printWriter.println(timestamp + " - INFO: " + info);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void logDebug(String debug) {
        try (FileWriter fileWriter = new FileWriter(LOG_FILE, true);
             PrintWriter printWriter = new PrintWriter(fileWriter)) {
            
            String timestamp = LocalDateTime.now().format(DATE_FORMATTER);
            printWriter.println(timestamp + " - DEBUG: " + debug);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        EventChronicle.logEvent("Application started");
        EventChronicle.logInfo("This is an info message");
        EventChronicle.logDebug("This is a debug message");
        EventChronicle.logError("This is an error message");
    }
}
