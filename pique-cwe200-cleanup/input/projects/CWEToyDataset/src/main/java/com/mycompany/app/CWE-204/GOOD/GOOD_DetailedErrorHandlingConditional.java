public class GOOD_DetailedErrorHandlingConditional {
    public void processRequest(String userInput) {
        if (userInput != null) {
            if (!userInput.isEmpty()) {
                if (!userInput.equals("admin")) {
                    System.out.println("Processing input: " + userInput);
                } else {
                    System.out.println("Error");
                }
            } else {
                System.out.println("Error");
            }
        } else {
            System.out.println("Error");
        }
    }
}
