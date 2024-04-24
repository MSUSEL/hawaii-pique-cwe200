import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GOOD_RedactedLogging {
    private static final Logger logger = LoggerFactory.getLogger(GOOD_RedactedLogging.class);

    public static void main(String[] args) {
        String sensitiveInfo = "Sensitive information here";
        // Secure: Redacting sensitive information before logging
        logger.info("Processing information: {}", redactSensitiveInfo(sensitiveInfo));
    }

    private static String redactSensitiveInfo(String info) {
        // Redaction logic, replacing sensitive information with placeholders
        return info.replaceAll(".", "*");
    }
}
