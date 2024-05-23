public class BAD_MedicalRecordSystemLookup {
    public void retrieveRecord(String patientId) {
        try {
            // Retrieve medical record logic
            throw new Exception("Medical record retrieval failed for patient ID: " + patientId);
        } catch (Exception e) {
            System.err.println("Medical record error: " + e.getMessage());
            throw new SecurityException("Error retrieving medical records, patient ID involved: " + patientId, e);
        }
    }

    public static void main(String[] args) {
        new BAD_MedicalRecordSystemLookup().retrieveRecord("patient_112233");
    }
}
