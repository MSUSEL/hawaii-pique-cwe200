import java.io.IOException;
import java.util.Properties;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import java.io.InputStream;

// This example demonstrates the secure loading of configuration properties from an external file.
public class GOOD_ConfigurationLoader {
    private ResourceLoader resourceLoader;

    public void Secure_ConfigurationLoader(ResourceLoader resourceLoader) {
        this.resourceLoader = resourceLoader;
    }

    public Properties loadProperties(String resourcePath) throws IOException {
        Resource resource = resourceLoader.getResource(resourcePath);
        Properties props = new Properties();
        try (InputStream is = (InputStream) resource.getInputStream()) {
            props.load(is);
        }
        return props;
    }
}
