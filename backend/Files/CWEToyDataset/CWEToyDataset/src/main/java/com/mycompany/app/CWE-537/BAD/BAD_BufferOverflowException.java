import java.nio.*;

public class BAD_BufferOverflowException { 
    private ByteBuffer dataBuffer;

    public BAD_BufferOverflowException(int bufferSize) {
        dataBuffer = ByteBuffer.allocate(bufferSize); // Allocate a fixed-size buffer for secure data
    }

    public void allocateSpace(byte[] sensitiveData) {
        try {
            dataBuffer.put(sensitiveData);
            System.out.println("Secure data stored successfully.");
        } catch (BufferOverflowException e) {
            System.err.println("Buffer overflow occurred while attempting to store secure data: " + new String(sensitiveData));
        } finally {
            dataBuffer.clear();
        }
    }

    public static void main(String[] args) {
        BAD_BufferOverflowException buffer = new BAD_BufferOverflowException(256); // Small buffer size for demonstration
        // Example of sensitive data: username and password details
        byte[] exampleSecureData = "User:Admin, Password:AdminPass123!".getBytes();
        buffer.allocateSpace(exampleSecureData); // This might cause a buffer overflow due to large data size
    }
}
