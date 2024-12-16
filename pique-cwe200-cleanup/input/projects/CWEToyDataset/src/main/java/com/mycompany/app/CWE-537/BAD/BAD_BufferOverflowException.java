import java.nio.*;

public class BAD_BufferOverflowException { 
    private ByteBuffer dataBuffer;

    public BAD_BufferOverflowException(int bufferSize) throws RuntimeException{
        dataBuffer = ByteBuffer.allocate(bufferSize);
    }

    public void allocateSpace(byte[] AWSKey) {
        try{
            dataBuffer.put(AWSKey);
        }
        catch (RuntimeException e){
            System.out.println("Buffer overflow detected, failed to save AWS Key" + AWSKey);
        }
    }
    public static void main(String[] args) {
        BAD_BufferOverflowException buffer = new BAD_BufferOverflowException(256); // Small buffer size for demonstration
        byte[] AWSKey = "ABCED123".getBytes();
        buffer.allocateSpace(AWSKey); // This might cause a buffer overflow due to large data size
    }
}


