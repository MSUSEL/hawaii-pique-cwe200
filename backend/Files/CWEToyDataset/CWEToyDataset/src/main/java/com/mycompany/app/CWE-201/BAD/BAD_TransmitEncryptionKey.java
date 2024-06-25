import java.io.OutputStream;
import java.net.Socket;

public class BAD_TransmitEncryptionKey {

    public void transmitKey(String host, int port, byte[] encryptionKey) {
        try (Socket socket = new Socket(host, port);
             OutputStream out = socket.getOutputStream()) {
            out.write(encryptionKey);
            out.flush();
            System.out.println("Encryption key sent to the server.");
        } catch (Exception e) {
            System.err.println("Failed to send encryption key: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        new BAD_TransmitEncryptionKey().transmitKey("example.com", 8080, "secretKey".getBytes());
    }
}
