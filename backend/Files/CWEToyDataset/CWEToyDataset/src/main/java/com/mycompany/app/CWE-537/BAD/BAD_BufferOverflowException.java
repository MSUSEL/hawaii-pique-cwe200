import java.nio.*;

public class BAD_BufferOverflowException { 
    private ByteBuffer dataBuffer;

    public BAD_BufferOverflowException(int bufferSize) {
        dataBuffer = ByteBuffer.allocate(bufferSize); // Allocate a fixed-size buffer for secure data
    }

    public void allocateSpace(byte[] AWSKey) {
        
        if (AWSKey.length > dataBuffer.capacity()) {
            throw new RuntimeException("Buffer overflow detected, failed to save AWS Key" + AWSKey); // Throw an exception if the data size exceeds the buffer capacity
        }
        else{
            dataBuffer.put(AWSKey);
        }
    }
    public static void main(String[] args) {
        BAD_BufferOverflowException buffer = new BAD_BufferOverflowException(256); // Small buffer size for demonstration
        // Example of sensitive data: username and password details
        byte[] AWSKey = "ABCED123".getBytes();
        buffer.allocateSpace(AWSKey); // This might cause a buffer overflow due to large data size
    }
}


