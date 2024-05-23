import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

// This example demonstrates the secure loading of configuration properties from an external file. So, that the sensitive information is not exposed.
public class GOOD_PropertyLoader {
    public Properties loadProperties(String filePath) throws IOException {
        Properties props = new Properties();
        try (FileInputStream fis = new FileInputStream(filePath)) {
            props.load(fis);
        }
        return props;
    }
}
