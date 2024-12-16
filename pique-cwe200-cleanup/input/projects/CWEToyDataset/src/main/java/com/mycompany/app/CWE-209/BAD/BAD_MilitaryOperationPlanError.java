// Define a custom exception
class OperationPlanAlreadyInitializedException extends Exception {
    public OperationPlanAlreadyInitializedException(String message) {
        super(message);
    }
}

public class BAD_MilitaryOperationPlanError {
    public void dispatchUnits(String operationPlan) {
        try {
            if (!operationPlan.isEmpty()) {
                throw new OperationPlanAlreadyInitializedException(
                    "Operation plan already initialized: " + operationPlan
                );
            }
            // Pretend to coordinate military units
            System.out.println("Units dispatched successfully.");
        } catch (OperationPlanAlreadyInitializedException e) {
            logError(e.getMessage());
        }
    }

    private void logError(String errorMessage) {
        // Log the error securely
        System.err.println("Error: " + errorMessage); // Replace with secure logging framework if needed
    }

    public static void main(String[] args) {
        new BAD_MilitaryOperationPlanError().dispatchUnits("Operation Red Dawn at 0400 hours; Secure the target zone.");
    }
}
