import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

public class GOOD_PropertyLoader {
    public Properties loadProperties(String filePath) throws IOException {
        Properties props = new Properties();
        try (FileInputStream fis = new FileInputStream(filePath)) {
            props.load(fis);
        }
        return props;
    }
}
