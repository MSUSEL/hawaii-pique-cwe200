export const commentsPrompt = `
Sensitive comments are those that have sensitive information such as passwords, API keys, personal data, or internal URLs.

### Examples of sensitive comments (These are just examples, you should look for more than these):
// Password: 123456
// Default admin password is admin123
// Use this temporary password: tempPass123!

// API key: 987654321
// API_KEY="sk_test_BQokikJOvBiI2HlWgH4olfQ2"
// Token for external API: abcd1234xyz


// Endpoint for internal services: https://api.internal.company.com
// Connect to internal database: jdbc:mysql://internal-db.example.com:3306/db

// Personal Information: John Doe, 123-45-6789
// Employee details: Name: Jane Doe, SSN: 987-65-4321
// Contact info: Email: john.doe@gmail.com, Phone: (555) 555-1234

// sensitiveData = "123456"
// The API token is 123
// Subnet mask: 255.255.255.0, Gateway: 192.168.1.1
// The server IP address is statically set to 192.168.1.100, which is not routed externally

// Current authentication can be bypassed using default admin account
// FIXME: calling this with more than 30 args kills the JDBC server


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
// encrypted password
// get scrambled password and store it encrypted
// the password in plain text
// Internal URL: https://example.com
// Contact info: Email: john.doe@example.com


### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1) Where it says "commentName1" and "commentDescription1" you should replace with the actual name and description of the sensitive comment.
2) Provide a JSON response for each file that matches the format below. 
  A) The "name" field should be the whole comment just how it is provided.
  B) The "isSensitive" field should be "yes" if you think this is a sensitive comment, and "no" if you don't think it is a sensitive comment.
  C) The "reason" field should describe the type of sensitive information found.
3) Do not include any \ in the name or description fields as this will cause the JSON to be invalid. For example, "-----BEGIN PRIVATE KEY-----\nMIICeAIBADANBgkqhkiG9w0BAQEFAASC...\n-----END PRIVATE KEY-----" should be written as "-----BEGIN PRIVATE KEY-----nMIICeAIBADANBgkqhkiG9w0BAQEFAASC...n-----END PRIVATE
{
  "files": [
    {
      "fileName": "FileName1.java",
      "comments": [
        {
          "name": "The whole comment",
          "isSensitive": "yes/no",
          "reason": "reason for the decision"
        }
      ]
    }
  ]
}

`
