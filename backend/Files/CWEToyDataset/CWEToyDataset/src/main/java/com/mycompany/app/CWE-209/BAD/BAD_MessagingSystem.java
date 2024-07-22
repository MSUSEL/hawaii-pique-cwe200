public class BAD_MessagingSystem {
    public void sendMessage(String userId, String messageContent) {
        try {
            throw new Exception("Message sending failed for user " + userId + " with content: " + messageContent);
        } catch (Exception e) {
            System.err.println("Messaging error: " + e.getMessage());
            throw new RuntimeException("Failed to send message, exposing sensitive content and user details.", e);
        }
    }

    public static void main(String[] args) {
        new BAD_MessagingSystem().sendMessage("user42", "Your secret PIN is 1234.");
    }
}
