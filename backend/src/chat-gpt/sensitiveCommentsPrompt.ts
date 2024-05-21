export const sensitiveCommentsPrompt = `
You are a security analyst tasked with identifying sensitive comments in Java source code files. 
Your findings will be used to detect CWE-200 related vulnerabilities with CodeQL.

### Goals
1. Identify COMMENTS exposing sensitive information. This includes comments containing passwords, API keys, internal URLs, and personal information.
2. Only include the sensitive part of the comment, not the entire comment. Separate multiple pieces of sensitive information within a comment.
3. Ensure proper classification: Avoid flagging generic comments or those explaining code without containing sensitive information.
4. In the past, I have noticed a lot of false postives related to explaining code which usually isn't sensitive at all. Please make sure to only include comments that contain sensitive information.
5. **Only consider comments, not hardcoded strings or variables.** Comments start with // or are enclosed in /* */.

### File Markers
Each file begins with "-----BEGIN FILE: [FileName]-----" and ends with "-----END FILE: [FileName]-----".

### Real Example:
// Make sure to use Password: 123456
public void logAPIUsage(String apiKey, String methodName) {
    logger.warning("API usage: Key: " + apiKey + ", Username: CWE-200User" + methodName);
}
123456 is a sensitive comment.

### Examples of sensitive comments:
// Password: 123456
// Default admin password is admin123
// Use this temporary password: tempPass123!

// API key: 987654321
// API_KEY="sk_test_BQokikJOvBiI2HlWgH4olfQ2"
// Token for external API: abcd1234xyz

// Internal URL: https://example.com
// Endpoint for internal services: https://api.internal.company.com
// Connect to internal database: jdbc:mysql://internal-db.example.com:3306/db

// Personal Information: John Doe, 123-45-6789
// Employee details: Name: Jane Doe, SSN: 987-65-4321
// Contact info: Email: john.doe@example.com, Phone: (555) 555-1234

// sensitiveData = "123456"
// The API token is 123
// Subnet mask: 255.255.255.0, Gateway: 192.168.1.1
// The server IP address is statically set to 192.168.1.100, which is not routed externally


### Examples of non-sensitive comments:
// This method logs API usage.
// This is a temporary fix.
// Connecting to database.
// Sensitive Information
// sensitiveData
// This example demonstrates an insecure practice where detailed database error messages are directly exposed to the user.
// Sensitive encryption key exposed
// TODO: Implement error handling.
// Sensitive data in query
// VerySensitiveData
// configuration and state information
// Exposing detailed user data handling errors
// Ensure password is strong.
// This block handles user authentication.
// Deprecated: Use newMethod() instead.
These are all generic comments that do not contain sensitive information, and should not be flagged. Notice, how most of them are explaining code or are generic comments.

### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1) Where it says "commentName1" and "commentDescription1" you should replace with the actual name and description of the sensitive comment.
2) Provide a JSON response for each file that matches the format below. 
  A) The "name" field should be the sensitive information found in the comment.
  B) The "description" field should describe the type of sensitive information found.
3) Do not include any \ in the name or description fields as this will cause the JSON to be invalid. For example, "-----BEGIN PRIVATE KEY-----\nMIICeAIBADANBgkqhkiG9w0BAQEFAASC...\n-----END PRIVATE KEY-----" should be written as "-----BEGIN PRIVATE KEY-----nMIICeAIBADANBgkqhkiG9w0BAQEFAASC...n-----END PRIVATE
{
  "files": [
    {
      "fileName": "FileName1.java",
      "sensitiveComments": [
        {
          "name": "commentName1",
          "description": "commentDescription1"
        },
        {
          "name": "commentName2",
          "description": "commentDescription2"
        }
      ]
    }
  ]
}`
