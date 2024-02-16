import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Map;

public class BAD_SystemPropertiesLog {
    private static final Logger logger = LoggerFactory.getLogger(BAD_SystemPropertiesLog.class);

    public static void main(String[] args) {
        // Insecure: Logging all system properties, including potentially sensitive ones
        for (Map.Entry<Object, Object> prop : System.getProperties().entrySet()) {
            logger.info(prop.getKey() + ": " + prop.getValue());
        }
    }
}
