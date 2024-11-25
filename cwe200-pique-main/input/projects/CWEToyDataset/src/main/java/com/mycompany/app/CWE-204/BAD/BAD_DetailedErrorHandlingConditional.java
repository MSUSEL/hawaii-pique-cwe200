public class BAD_DetailedErrorHandlingConditional {
    public void processRequest(String userInput) {
        if (userInput != null) {
            if (!userInput.isEmpty()) {
                if (!userInput.equals("admin")) {
                    System.out.println("Processing input: " + userInput);
                } else {
                    System.out.println("Error: 'admin' is a reserved keyword");
                }
            } else {
                System.out.println("Error: Input cannot be empty");
            }
        } else {
            System.out.println("Error: Input cannot be null");
        }
    }
}
