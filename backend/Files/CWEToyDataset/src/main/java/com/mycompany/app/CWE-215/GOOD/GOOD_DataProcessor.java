public class GOOD_DataProcessor {
    private static final boolean DATA_PROCESSING_DEBUG = Boolean.parseBoolean(System.getProperty("DATA_PROCESSING_DEBUG", "false"));

    public static void main(String[] args) {
        try {
            processData("sensitiveDataString");
        } catch (DataProcessingException e) {
            if (DATA_PROCESSING_DEBUG) {
                System.err.println("DEBUG: Error processing data.");
            }
        }
    }

    private static void processData(String data) throws DataProcessingException {
        // Simulating a processing error
        throw new DataProcessingException("Processing failed at index: 10", data.substring(5, 15));
    }

    private static class DataProcessingException extends Exception {
        private String dataSnippet;

        public DataProcessingException(String message, String dataSnippet) {
            super(message);
            this.dataSnippet = dataSnippet;
        }

        public String getDataSnippet() {
            return dataSnippet;
        }
    }
}
