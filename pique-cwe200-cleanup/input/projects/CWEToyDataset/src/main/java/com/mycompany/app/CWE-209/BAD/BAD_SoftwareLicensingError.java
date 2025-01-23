public class BAD_SoftwareLicensingError {
    public void checkSoftwareLicense(String licenseKey) {
        try {
            if (licenseKey == null || licenseKey.isEmpty()) {
                throw new Exception("License key is missing: " + licenseKey);
            }
            // License checking logic
        } catch (Exception e) {
            System.err.println("Software license validation failed: " + e.getMessage());
            throw new RuntimeException("Critical licensing error, contact support with code: " + licenseKey, e);
        }
    }

    public static void main(String[] args) {
        new BAD_SoftwareLicensingError().checkSoftwareLicense("");
    }
}
