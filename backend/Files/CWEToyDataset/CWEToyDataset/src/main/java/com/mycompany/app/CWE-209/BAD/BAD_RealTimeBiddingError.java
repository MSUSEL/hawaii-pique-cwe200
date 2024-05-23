public class BAD_RealTimeBiddingError {
    public void processBid(String bidData, String sessionId) {
        try {
            if (bidData.isEmpty()) {
                throw new Exception("Empty bid data received for session: " + sessionId);
            }
            // Bid processing logic
        } catch (Exception e) {
            String errorDetails = "Error processing bid for session " + sessionId + ": " + e.getMessage();
            System.err.println(errorDetails);
            throw new RuntimeException("Failed to process bid, contact tech support with this session ID.", e);
        }
    }

    public static void main(String[] args) {
        new BAD_RealTimeBiddingError().processBid("", "Session1234");
    }
}
