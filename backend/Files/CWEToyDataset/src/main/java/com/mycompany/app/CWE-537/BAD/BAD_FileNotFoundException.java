import java.io.*;
import java.util.logging.*;

public class BAD_FileNotFoundException {
    private static final Logger logger = Logger.getLogger("FileProcessingLogger");

    public void processFile(String filePath) {
        try {
            new FileReader(filePath);
        } catch (FileNotFoundException e) {
            logger.warning("File processing failed. File path: " + filePath + ". Error: " + e.toString());
            throw new RuntimeException("File not found at " + filePath);
        }
    }
}
