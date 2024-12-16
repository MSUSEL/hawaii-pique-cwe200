import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;

public class BAD_MedicalRecordSystemLookup {
    public void retrieveRecord(String patientId) {
        Connection connection = null;
        PreparedStatement statement = null;
        ResultSet resultSet = null;

        try {
            // Establish database connection
            connection = DriverManager.getConnection("jdbc:mysql://hawaiihealth:3306/medical_records", "user", "password");

            // Prepare SQL query to retrieve medical record
            String sql = "SELECT * FROM medical_records WHERE patient_id = ?";
            statement = connection.prepareStatement(sql);
            statement.setString(1, patientId);

            // Execute query and process result set
            resultSet = statement.executeQuery();
            if (resultSet.next()) {
                System.out.println("Medical record for patient ID: " + patientId + " retrieved successfully.");
            } else {
                throw new Exception("No medical record found for patient ID: " + patientId);
            }
        } catch (Exception e) {
            System.err.println("Medical record error: " + e.getMessage());
        } finally {
            // Close resources
            try {
                if (resultSet != null) resultSet.close();
                if (statement != null) statement.close();
                if (connection != null) connection.close();
            } catch (Exception e) {
                System.err.println("Error closing resources: " + e.getMessage());
            }
        }
    }

    public static void main(String[] args) {
        new BAD_MedicalRecordSystemLookup().retrieveRecord("patient_112233");
    }
}