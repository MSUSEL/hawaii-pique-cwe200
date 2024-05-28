public class BAD_DetailedErrorHandlingConditional {
    public void processRequest(String userInput) {
        if (userInput == null) {
            System.out.println("Error: Input cannot be null");
        } else {
            if (userInput.isEmpty()) {
                System.out.println("Error: Input cannot be empty");
            } else {
                if (userInput.equals("admin")) {
                    System.out.println("Error: 'admin' is a reserved keyword");
                } else {
                    System.out.println("Processing input: " + userInput);
                }
            }
        }
    }
}
