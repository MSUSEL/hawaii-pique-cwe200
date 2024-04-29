import java.nio.*;
import java.util.logging.*;

public class BAD_BufferOverflowException {
    private ByteBuffer dataBuffer;
    private static final Logger logger = Logger.getLogger("SecurityDataLogger");

    public BAD_BufferOverflowException(int bufferSize) {
        dataBuffer = ByteBuffer.allocate(bufferSize); // Allocate a fixed-size buffer for secure data
    }

    public void logSecureData(byte[] sensitiveData) {
        try {
            dataBuffer.put(sensitiveData); // Attempt to store sensitive data in the buffer
            System.out.println("Secure data logged successfully.");
        } catch (BufferOverflowException e) {
            // Log includes sensitive data attempting to be stored, which exposes it when buffer overflow occurs
            logger.severe("Buffer overflow when trying to log secure data: '" + new String(sensitiveData) + "'. Buffer capacity: " + dataBuffer.capacity() + " bytes.");
            throw new BufferOverflowException();
        } finally {
            dataBuffer.clear(); // Clear buffer for the next use
        }
    }

    public static void main(String[] args) {
        BAD_BufferOverflowException dataLogger = new BAD_BufferOverflowException(256); // Small buffer size for demonstration
        // Example of sensitive data: username and password details
        byte[] exampleSecureData = "User:Admin, Password:AdminPass123!".getBytes();
        dataLogger.logSecureData(exampleSecureData); // This might cause a buffer overflow due to large data size
    }
}
