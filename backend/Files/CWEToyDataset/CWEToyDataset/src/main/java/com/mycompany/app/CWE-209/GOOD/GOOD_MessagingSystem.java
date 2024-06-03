public class GOOD_MessagingSystem {
    public void sendMessage(String userId, String messageContent) {
        try {
            // Assume a failure in sending the message
            throw new Exception("Messaging service interruption.");
        } catch (Exception e) {
            System.err.println("Messaging service error.");
            throw new RuntimeException("Messaging service currently unavailable.");
        }
    }

    public static void main(String[] args) {
        new GOOD_MessagingSystem().sendMessage("user42", "Confidential content.");
    }
}