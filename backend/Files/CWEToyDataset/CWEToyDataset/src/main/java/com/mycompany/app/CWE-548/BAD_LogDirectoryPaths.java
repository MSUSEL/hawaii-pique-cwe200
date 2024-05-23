import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.LogManager;
import java.io.File;

// BAD: Application logs include sensitive directory paths and file details
public class BAD_LogDirectoryPaths {
    private static final Logger logger = LogManager.getLogger(BAD_LogDirectoryPaths.class);

    public void processFilesInDirectory(String directoryPath) {
        File directory = new File(directoryPath);
        
        // Insecurely logs the directory path
        logger.info("Processing files in directory: " + directoryPath);

        // Check if directory exists and is indeed a directory
        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles();

            if (files != null) {
                for (File file : files) {
                    // Simulate processing each file (e.g., reading, moving, or updating files)
                    logger.info("Processing file: " + file.getName()); // Insecurely logs file names
                    
                    // Example of further logging that could reveal sensitive information
                    if (file.isFile()) {
                        logger.debug("File size (bytes): " + file.length());
                    } else if (file.isDirectory()) {
                        // Further directory logging could inadvertently expose more sensitive information
                        logger.debug("Encountered nested directory: " + file.getName());
                    }
                    
                    // Placeholder for actual file processing logic (e.g., parsing content, moving file)
                }
            } else {
                logger.warn("No files found in directory: " + directoryPath);
            }
        } else {
            logger.error("The path provided does not exist or is not a directory: " + directoryPath);
        }
    }

    public static void main(String[] args) {
        new BAD_LogDirectoryPaths().processFilesInDirectory("/path/to/sensitive/data");
    }
}
