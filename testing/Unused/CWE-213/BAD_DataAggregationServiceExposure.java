public class BAD_DataAggregationServiceExposure {
    // This class represents a data aggregation service that compiles and exposes summaries of user activities across various platforms.
    // The service inadvertently exposes sensitive user data due to its broad aggregation and sharing policies.

    public String aggregateUserData(String userId) {
        String aggregatedData = fetchDataFromSources(userId);
        if (aggregatedData.contains("sensitive")) {
            // Insecure: The service does not filter out sensitive information from the aggregated data before sharing.
            // It assumes all aggregated data is non-sensitive according to its own policy, overlooking user privacy expectations.
            return "Aggregated Data for " + userId + ": " + aggregatedData;
        }
        return "No data available for " + userId;
    }

    private String fetchDataFromSources(String userId) {
        // Simulate fetching data from multiple sources, some of which might include sensitive information.
        // For demonstration, returning a string that contains what might be considered sensitive data.
        return "User activities: [sensitive medical data, financial transactions]";
    }

    public static void main(String[] args) {
        BAD_DataAggregationServiceExposure service = new BAD_DataAggregationServiceExposure();
        System.out.println(service.aggregateUserData("user123"));
    }
}
