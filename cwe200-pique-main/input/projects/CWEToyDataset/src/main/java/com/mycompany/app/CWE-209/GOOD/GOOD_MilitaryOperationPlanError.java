public class GOOD_MilitaryOperationPlanError {
    public void dispatchUnits(String operationPlan) {
        try {
            if (operationPlan.isEmpty()) {
                throw new SecurityException("Operation plan required.");
            }
            // Pretend to coordinate military units
        } catch (SecurityException e) {
            System.err.println("Operation cannot be initiated without a plan.");
        }
    }

    public static void main(String[] args) {
        new GOOD_MilitaryOperationPlanError().dispatchUnits("");
    }
}