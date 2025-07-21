
import java

string suspicious(string fileName) {
fileName = "GOOD_SecureKeyManagement.java" and result = ["ENCRYPTION_KEY"] or
fileName = "GOOD_GeneralizedAuthenticationFeedback.java" and result = ["username", "password"] or
fileName = "BAD_VerboseDatabaseErrorMessage.java" and result = ["connection"] or
fileName = "BAD_DataAggregationServiceExposure.java" and result = ["aggregatedData"] or
fileName = "BAD_DatabaseSchemaExposure.java" and result = ["connection", "statement"] or
fileName = "BAD_LoggingSensitiveUserInfo.java" and result = ["username", "password"] or
fileName = "GOOD_TimingAttackAgainstHeader.java" and result = ["Key"] or
fileName = "BAD_DatabaseConnectionError.java" and result = ["url", "user", "password"] or
fileName = "BAD_FileReadSensitiveError.java" and result = ["filePath"] or
fileName = "GOOD_SecureStorageForSensitiveInfo.java" and result = ["secretPassword"] or
fileName = "BAD_DebugLoggingSensitiveInfo.java" and result = ["username", "password"] or
fileName = "BAD_DebugOutputWithCredentials.java" and result = ["username", "password"] or
fileName = "BAD_ExceptionDebuggingWithSensitiveData.java" and result = ["creditCardNumber"] or
fileName = "GOOD_EncryptDataBeforeTransmission.java" and result = ["KEY", "APIKey"] or
fileName = "GOOD_UseHttpsForSensitiveData.java" and result = ["sensitiveData"] or
fileName = "BAD_AuthenticationTiming.java" and result = ["validUsername", "attemptUsername", "attemptPassword"] or
fileName = "BAD_ConditionalErrorHandling.java" and result = ["userData", "username"] or
fileName = "BAD_LoginResponseDiscrepancy.java" and result = ["username", "password"] or
fileName = "BAD_CommandLineToolSensitiveData.java" and result = ["sensitiveFilePath"] or
fileName = "BAD_EnvVarExposure.java" and result = ["API_KEY"] or
fileName = "BAD_ProcessInvocationWithArgs.java" and result = ["pass"] or
fileName = "GOOD_RedactedLogging.java" and result = ["sensitiveInfo"] or
fileName = "BAD_ApiKeyExposureTestConfig.java" and result = ["apiKey"] or
fileName = "BAD_EmbeddedSensitiveDataInTests.java" and result = ["personalEmail", "personalPhone"] or
fileName = "BAD_HardcodedCredentialsTest.java" and result = ["username", "password"] or
fileName = "BAD_CreditCardSubmissionGET.java" and result = ["cardNumber", "expiryDate", "cvv"] or
fileName = "BAD_HealthInfoSubmissionGET.java" and result = ["patientId", "symptoms", "doctorId"] or
fileName = "BAD_PasswordChangeGET.java" and result = ["username", "oldPassword", "newPassword"] or
fileName = "GOOD_DocumentAccessServlet.java" and result = ["requestedFileName", "fileDirectory", "filePath"] or
fileName = "GOOD_SecureBackendFileProcessingServlet.java" and result = ["tempFilesPath", "tempDirectory"] or
fileName = "BAD_VerboseSQLException.java" and result = ["connection"] or
fileName = "BAD_LoggingCredentials.java" and result = ["username", "password"] or
fileName = "BAD_LogPaymentTransactions.java" and result = ["fullName", "creditCardNumber", "amount"] or
fileName = "BAD_StoreDbConnectionInfo.java" and result = ["dbUrl", "dbUser", "dbPassword"] or
fileName = "GOOD_EnvironmentVariablesForSensitiveDataTest.java" and result = ["username", "password"] or
fileName = "GOOD_ExternalizedSensitiveInfoConfigTest.java" and result = ["apiKey"] or
fileName = "BAD_LogDbConnectionDetails.java" and result = ["dbUrl", "user", "password"] or
fileName = "BAD_LogPaymentInformation.java" and result = ["creditCardNumber", "expiryDate", "cvv"] or
fileName = "BAD_LogUserCredentials.java" and result = ["username", "password"] or
fileName = "GOOD_EnhancedDebugWithoutSensitiveData.java" and result = ["userId"] or
fileName = "GOOD_SecureDebuggingPractices.java" and result = ["username", "password"] or
fileName = "GOOD_SecureConfigStorage.java" and result = ["dbUrl"] or
fileName = "GOOD_SecureLogging.java" and result = ["username"] or
fileName = "BAD_ApiKeyEmbedded.java" and result = ["SERVICE_API_KEY"] or
fileName = "BAD_HardcodedCredentials.java" and result = ["DATABASE_URL", "DATABASE_USER", "DATABASE_PASSWORD"] or
fileName = "BAD_HardcodedSecretInConfig.java" and result = ["apiKey", "encryptionKey"] or
fileName = "GOOD_EncapsulatedSecurityContext.java" and result = ["command", "errorOutput", "sanitizedError"] or
fileName = "GOOD_SecureShellCommandHandling.java" and result = ["command"] or
fileName = "BAD_DbConnectionErrorServlet.java" and result = ["conn"] or
fileName = "BAD_ExternalServiceErrorServlet.java" and result = ["url", "connection"] or
fileName = "BAD_FileReadErrorServlet.java" and result = ["filename", "file"] or
fileName = "GOOD_SecureLoggingWithCodes.java" and result = ["accountNumber"] or
fileName = "BAD_ExcessiveDatabaseExceptionDetails.java" and result = ["user", "pass"] or
fileName = "BAD_WebServiceExceptionExposure.java" and result = ["endpointUrl"] or
fileName = "GOOD_ConsistentAuthenticationTiming.java" and result = ["VALID_USERNAME", "VALID_PASSWORD"] or
fileName = "GOOD_UniformLoginResponse.java" and result = ["username", "password"] or
fileName = "BAD_ConditionalContentDelivery.java" and result = ["userRole"] or
fileName = "BAD_DifferentialErrorHandling.java" and result = ["data"] or
fileName = "BAD_UserEnumerationLogin.java" and result = ["validUsername", "validPassword", "username", "password"] or
fileName = "GOOD_FilterSensitiveInfoFromLogs.java" and result = ["username", "password"] or
fileName = "BAD_MisconfiguredPermissions.java" and result = ["scriptPath"] or
fileName = "BAD_SensitiveInfoExposureViaShellError.java" and result = ["command"] or
fileName = "BAD_ShellCommandExposure.java" and result = ["command"] or
fileName = "GOOD_FixedResponseTiming.java" and result = ["VALID_USERNAME", "VALID_PASSWORD"] or
fileName = "GOOD_UniformErrorResponses.java" and result = ["VALID_USERNAME", "VALID_PASSWORD"] or
fileName = "BAD_TimingAttackAgainstHeader.java" and result = ["Key"] or
fileName = "BAD_VulnerableStringComparison.java" and result = ["secretPassword"] or
fileName = "GOOD_ExternalizeSensitiveConfig.java" and result = ["databaseUrl", "databaseUser", "databasePassword"] or
fileName = "GOOD_UseEnvVarsForCredentials.java" and result = ["databaseUrl", "databaseUser", "databasePassword"] or
fileName = "BAD_DocumentDirectoryListingServlet.java" and result = ["documentPath"] or
fileName = "BAD_InsecureDirectoryListingServlet.java" and result = ["path"] or
fileName = "BAD_LogDirectoryPaths.java" and result = ["directoryPath"] or
fileName = "BAD_TempDirUsageVulnerable.java" and result = ["temp1", "temp2", "temp3", "systemTempDir"] or
fileName = "GOOD_TempDirUsageSafe.java" and result = ["temp1", "temp2", "tempChildFile"] or
fileName = "BAD_LogSensitiveInfoBeforeSending.java" and result = ["creditCardNumber", "payload"] or
fileName = "BAD_SendSensitiveDataViaEmail.java" and result = ["username", "password", "smtpHost", "smtpPort", "sensitiveContent"] or
fileName = "BAD_SendSensitiveInfoInGetRequest.java" and result = ["username", "password"]
}

string suspiciousStrings(string fileName) {
fileName = "BAD_ConfigurationPathException.java" and result = ["/etc/myapp/config.properties"] or
fileName = "BAD_FileAccessErrorExposure.java" and result = ["/etc/secret-config.properties"] or
fileName = "BAD_VerboseDatabaseErrorMessage.java" and result = ["jdbc:mysql://localhost:3306/myapp", "user", "pass"] or
fileName = "BAD_DataAggregationServiceExposure.java" and result = ["sensitive medical data", "financial transactions"] or
fileName = "BAD_LoggingSensitiveUserInfo.java" and result = ["admin", "adminPass"] or
fileName = "GOOD_TimingAttackAgainstHeader.java" and result = ["X-Auth-Token", "secret"] or
fileName = "BAD_DetailedDatabaseErrorExposure.java" and result = ["SELECT * FROM sensitive_table", "Failed to execute query on 'sensitive_table': Column 'secret_info' does not exist"] or
fileName = "BAD_SensitivePathDisclosureInErrors.java" and result = ["/sensitive/config/secrets.txt", "Error: File not found - /sensitive/config/secrets.txt"] or
fileName = "BAD_VerboseAuthenticationFailure.java" and result = ["user", "guessme", "Authentication failed: user does not exist or password is incorrect."] or
fileName = "BAD_DatabaseConnectionError.java" and result = ["jdbc:mysql://localhost:3306/sensitiveDB", "sensitiveUser", "sensitivePassword"] or
fileName = "BAD_FileReadSensitiveError.java" and result = ["/etc/sensitive/config.txt"] or
fileName = "BAD_NetworkConfigError.java" and result = ["sensitive.internal.service"] or
fileName = "BAD_DebugLoggingSensitiveInfo.java" and result = ["defaultUser", "defaultPass"] or
fileName = "BAD_DebugOutputWithCredentials.java" and result = ["admin", "wrongPassword", "adminSecret"] or
fileName = "BAD_ExceptionDebuggingWithSensitiveData.java" and result = ["1234-5678-9012-3456"] or
fileName = "GOOD_EncryptDataBeforeTransmission.java" and result = ["Bar12345Bar12345", "The API token is 123"] or
fileName = "GOOD_UseHttpsForSensitiveData.java" and result = ["This is sensitive"] or
fileName = "BAD_ConditionalErrorHandling.java" and result = ["existingUser", "userDataIncludingSensitiveInfo"] or
fileName = "GOOD_ExplicitUserConsent.java" and result = ["User's private browsing data"] or
fileName = "GOOD_SanitizeLogs.java" and result = ["Sensitive user input: john.doe@example.com or 192.168.1.1 or 1234567812345678"] or
fileName = "BAD_CommandLineToolSensitiveData.java" and result = ["/path/to/sensitive/data.txt"] or
fileName = "BAD_EnvVarExposure.java" and result = ["secretApiKey"] or
fileName = "BAD_ProcessInvocationWithArgs.java" and result = ["helloWorld123", "someCommand --password="] or
fileName = "BAD_ApiKeyExposureTestConfig.java" and result = ["test-config.properties", "api.key"] or
fileName = "BAD_EmbeddedSensitiveDataInTests.java" and result = ["user@example.com", "555-0100"] or
fileName = "BAD_HardcodedCredentialsTest.java" and result = ["admin", "admin123"] or
fileName = "BAD_CreditCardSubmissionGET.java" and result = ["1234567890123456", "12/23", "123"] or
fileName = "BAD_HealthInfoSubmissionGET.java" and result = ["12345", "Fever, cough, and shortness of breath", "DrJohnDoe"] or
fileName = "BAD_PasswordChangeGET.java" and result = ["user2", "oldPass123", "newPass456"] or
fileName = "BAD_DetailedRuntimeExceptionServlet.java" and result = ["Failed to process due to invalid state in user profile"] or
fileName = "BAD_ExposedExceptionDetailsAPI.java" and result = ["Database connection error: Credentials for 'db_user' not valid or expired"] or
fileName = "BAD_LoggingCredentials.java" and result = ["admin", "password123"] or
fileName = "BAD_LogPaymentTransactions.java" and result = ["Alice Johnson", "1234567890123456", "99.99"] or
fileName = "BAD_StoreDbConnectionInfo.java" and result = ["jdbc:mysql://localhost:3306/sensitiveDB", "sensitiveUser", "sensitivePassword"] or
fileName = "BAD_LogDbConnectionDetails.java" and result = ["jdbc:mysql://localhost:3306/myapp", "admin", "admin123"] or
fileName = "BAD_LogPaymentInformation.java" and result = ["1234567890123456", "12/23", "123"] or
fileName = "BAD_LogUserCredentials.java" and result = ["admin", "adminPass", "user1", "pass1234"] or
fileName = "BAD_DetailedErrorServlet.java" and result = ["Failed to load configuration from /etc/app/config.properties"] or
fileName = "BAD_InsecureDebugMode.java" and result = ["Sensitive operation failed due to configuration error: /etc/app/config.secret"] or
fileName = "BAD_ApiKeyEmbedded.java" and result = ["ABCD1234EFGH5678"] or
fileName = "BAD_HardcodedCredentials.java" and result = ["admin", "password123"] or
fileName = "BAD_HardcodedSecretInConfig.java" and result = ["supersecretkey12345", "0123456789abcdef"] or
fileName = "BAD_DbConnectionErrorServlet.java" and result = ["jdbc:mysql://localhost:3306/myapp", "user", "pass"] or
fileName = "BAD_ExternalServiceErrorServlet.java" and result = ["https://api.example.com/data"] or
fileName = "BAD_FileReadErrorServlet.java" and result = ["/var/www/data/"] or
fileName = "BAD_ExcessiveDatabaseExceptionDetails.java" and result = ["jdbc:mysql://localhost:3306/nonexistentDB", "username = "] or
fileName = "BAD_VerboseLibraryExceptionHandling.java" and result = ["/secure/config/api_keys.config"] or
fileName = "BAD_WebServiceExceptionExposure.java" and result = ["http://example.com/nonexistent/service"] or
fileName = "GOOD_SecureLoggingPractices.java" and result = ["Updating user password", "Viewing general user profile information"] or
fileName = "BAD_MisconfiguredPermissions.java" and result = ["/path/to/sensitive/script.sh"] or
fileName = "BAD_SensitiveInfoExposureViaShellError.java" and result = ["somecommand --password=secret"] or
fileName = "BAD_ShellCommandExposure.java" and result = ["cp /path/to/sensitive/file /backup/location"] or
fileName = "BAD_LogSensitiveInfoBeforeSending.java" and result = ["creditCardNumber"] or
fileName = "BAD_SendSensitiveDataViaEmail.java" and result = ["sensitiveContent"] or
fileName = "BAD_SendSensitiveInfoInGetRequest.java" and result = ["username", "password"]
}



string suspiciousComments() {
  result = 
  [
    "This class represents a data aggregation service that compiles and exposes summaries of user activities across various platforms. The service inadvertently exposes sensitive user data due to its broad aggregation and sharing policies.","This example demonstrates an insecure practice where detailed database error messages are directly exposed to the user.","This example shows an insecure practice of including detailed file system paths in error messages when file access is denied or the file does not exist.","This class shows an insecure approach to handling authentication failures by providing detailed error messages.","ak_test_12345","AESKey123456789","admin","admin12345","jdbc:mysql://localhost:3306/production"
  ]
}    
    

    
class SensitiveVariable extends Variable {
  SensitiveVariable() {
    exists(File f | 
      f = this.getCompilationUnit().getFile() and
      this.getName().matches(suspicious(f.getBaseName()))
    )
  }
}

class SensitiveVariableExpr extends Expr {
  SensitiveVariableExpr() {
    exists(Variable v, File f | this = v.getAnAccess() and
      f = v.getCompilationUnit().getFile() and
      v.getName().matches(suspicious(f.getBaseName())) and
      not this instanceof CompileTimeConstantExpr
    )
  }
}

class SensitiveStringLiteral extends StringLiteral {
  SensitiveStringLiteral() {
    // Check for matches against the suspicious patterns
    exists(File f | 
      f = this.getCompilationUnit().getFile() and
      this.getValue().regexpMatch(suspiciousStrings(f.getBaseName()))    
      ) and
    not exists(MethodCall mc |
      mc.getAnArgument() = this and
      (
        mc.getMethod().hasName("getenv") or
        mc.getMethod().hasName("getParameter") or
        mc.getMethod().hasName("getProperty") 
      )
    )
  }   
}

class SensitiveComment extends StringLiteral {
  SensitiveComment() {
    // Check for matches against the suspicious patterns
    exists(File f | 
      f = this.getCompilationUnit().getFile() and
      this.getValue().regexpMatch(suspiciousComments())    
    ) 
  }   
}
