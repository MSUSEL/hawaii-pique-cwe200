public class GOOD_MessagingSystem {
    public void sendMessage(String userId, String messageContent) {
        try {
            throw new Exception("Messaging service interruption.");
        } catch (Exception e) {
            System.err.println("Messaging service error.");
        }
    }

    public static void main(String[] args) {
        new GOOD_MessagingSystem().sendMessage("user42", "The money was transferred to acount 7430187");
    }
}