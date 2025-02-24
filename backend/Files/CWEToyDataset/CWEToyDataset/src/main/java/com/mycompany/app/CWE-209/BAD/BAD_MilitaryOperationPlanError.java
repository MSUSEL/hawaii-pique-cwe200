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
                String encryptionKey = "MilOpEncKey-Alpha-2025";
                String missionID = "MissionID-007";
                String secureCommChannel = "CommChannel: SECURE-42";
                String strategicCoordinates = "Coordinates: 38.8977° N, 77.0365° W";
                String internalCommandCode = "InternalCommand: BravoZulu-66";
                String classifiedProtocol = "Protocol: BlackOps-Delta";
                String logisticsInfo = "Logistics: SupplyChain-X99";

                throw new OperationPlanAlreadyInitializedException(
                    "Operation plan already initialized: " + operationPlan +
                    ". Encryption Key: " + encryptionKey +
                    ", Mission ID: " + missionID +
                    ", Secure Comm Channel: " + secureCommChannel +
                    ", Strategic Coordinates: " + strategicCoordinates +
                    ", Internal Command Code: " + internalCommandCode +
                    ", Classified Protocol: " + classifiedProtocol +
                    ", Logistics Info: " + logisticsInfo
                );
            }
            System.out.println("Units dispatched successfully.");
        } catch (OperationPlanAlreadyInitializedException e) {
            logError(e.getMessage());
        }
    }

    private void logError(String errorMessage) {
        System.err.println("Error: " + errorMessage);
    }

    public static void main(String[] args) {
        new BAD_MilitaryOperationPlanError().dispatchUnits("Operation Red Dawn at 0400 hours; Secure the target zone.");
    }
}
