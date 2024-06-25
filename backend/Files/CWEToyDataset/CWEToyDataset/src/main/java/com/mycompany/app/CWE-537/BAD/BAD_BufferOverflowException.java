import java.nio.*;

public class BAD_BufferOverflowException { 
    private ByteBuffer dataBuffer;

    public BAD_BufferOverflowException(int bufferSize) {
        dataBuffer = ByteBuffer.allocate(bufferSize); // Allocate a fixed-size buffer for secure data
    }

    public void allocateSpace(byte[] sensitiveData) {
        
        if (sensitiveData.length > dataBuffer.capacity()) {
            System.err.println("Buffer overflow detected!" + sensitiveData);
            throw new BufferOverflowException(); // Throw an exception if the data size exceeds the buffer capacity
        }
        else{
            dataBuffer.put(sensitiveData);
        }
        

    }
    public static void main(String[] args) {
        BAD_BufferOverflowException buffer = new BAD_BufferOverflowException(256); // Small buffer size for demonstration
        // Example of sensitive data: username and password details
        byte[] exampleSecureData = "User:Admin, Password:AdminPass123!".getBytes();
        buffer.allocateSpace(exampleSecureData); // This might cause a buffer overflow due to large data size
    }
}


