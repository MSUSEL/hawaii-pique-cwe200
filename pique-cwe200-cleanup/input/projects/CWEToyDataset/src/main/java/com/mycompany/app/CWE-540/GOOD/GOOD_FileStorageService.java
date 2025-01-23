import java.nio.file.Path;
import java.nio.file.Paths;

public class GOOD_FileStorageService {
    private Path storagePath;

    public GOOD_FileStorageService() {
        // Load storage path from environment variable
        this.storagePath = Paths.get(System.getenv("STORAGE_PATH"));
    }

    public void saveFile(byte[] data, String fileName) {
        Path filePath = storagePath.resolve(fileName);
        System.out.println("Saving file to: " + filePath);
        // Assume this method would actually save the file to disk
    }
}
