public class BAD_MessagingSystem {
    public void sendMessage(String userId, String messageContent) {
        try {
            String encryptionKey = "MsgEncKey-Secret-123456";
            String internalMsgId = "MSG-00042-SECRET";
            String serverIp = "10.0.0.15";
            
            throw new Exception("Message sending failed for user " + userId + " with content: " + messageContent +
                                ". Internal Encryption Key: " + encryptionKey +
                                ", Internal Message ID: " + internalMsgId +
                                ", Server IP: " + serverIp);
        } catch (Exception e) {
            System.err.println("Messaging error: " + e.getMessage());
            throw new RuntimeException("Failed to send message, exposing sensitive content, encryption details, and server configuration.", e);
        }
    }

    public static void main(String[] args) {
        new BAD_MessagingSystem().sendMessage("user42", "Your PIN is 1234.");
    }
}
