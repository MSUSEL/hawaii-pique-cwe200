export const sensitiveVariablesPrompt = `
You are a security analyst tasked with identifying sensitive variables in Java source code files. 
Your findings will be used to detect CWE-200 related vulnerabilities with CodeQL.

### Goals
1. Identify VARIABLES exposing sensitive information. This includes variables related to passwords, API keys, internal URLs, database connections, personal information, etc.
2. Ensure proper classification: Avoid flagging generic variables or those without sensitive information.
3. In the past, I have noticed a lot of false positives related to generic variables which usually aren't sensitive. Please make sure to only include variables that contain sensitive information.
4. **Only consider variables, not hardcoded strings or comments.**

### File Markers
Each file begins with "-----BEGIN FILE: [FileName]-----" and ends with "-----END FILE: [FileName]-----".

### Real Example:
public class Example {
    private static final String PASSWORD = "123456";
    public void logAPIUsage(String apiKey, String methodName) {
        logger.warning("API usage: Key: " + apiKey + ", Username: CWE-200User" + methodName);
    }
}
PASSWORD is a sensitive variable.

### Examples of sensitive variables:
PASSWORD, API_KEY, INTERNAL_URL, PERSONAL_INFO, DB_CONNECTION_STRING, data, command, apikey, key, user, pass, file, 
deviceID, firmwareVersion, userID, password, username, token, secret, accessKey, privateKey, publicKey, certificate,
email, message, phone, address, city, state, zip, country, ssn, fullName, dateOfBirth, dob, DOB, ssnNumber, ssnLast4, ssnLast4Digits,
creditCard, cc, ccNumber, ccDigits, ccDigitsOnly, ccLast4, ccLast4Digits, ccExpiration, ccExpirationDate, ccSecurityCode,
patientId, patientNumber, patientName, patientDOB, patientSSN, patientSSNNumber, bankAccount, bankAccountNumber, bankAccountDigits,
licenseKey

### Examples of non-sensitive variables:
encryptedData, 

### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1) Where it says "variableName1" and "variableDescription1" you should replace with the actual name and description of the sensitive variable.
2) Provide a JSON response for each file that matches the format below. 
  A) The "name" field should be the sensitive information found in the variable.
  B) The "description" field should describe the type of sensitive information found.
{
  "files": [
    {
      "fileName": "FileName1.java",
      "sensitiveVariables": [
        {
          "name": "variableName1",
          "description": "variableDescription1"
        },
        {
          "name": "variableName2",
          "description": "variableDescription2"
        }
      ]
    }
  ]
}`
