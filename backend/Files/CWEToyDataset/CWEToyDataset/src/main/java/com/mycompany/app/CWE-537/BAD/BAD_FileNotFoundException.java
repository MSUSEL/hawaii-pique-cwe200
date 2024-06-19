import java.io.*;

public class BAD_FileNotFoundException {

    public void processFile(String filePath) throws FileNotFoundException {
        if (!isValidFilePath(filePath)) {
            throw new FileNotFoundException("File not found at " + filePath);
        } else {
            try {
                FileReader reader = new FileReader(filePath);
                reader.close();
            } catch (IOException e) {
                // Handle or log the IOException
                System.err.println("An I/O error occurred: " + e.getMessage());
            }
        }
    }

    private boolean isValidFilePath(String filePath) {
        // Check if the file path is valid
        return true;
    }
}
