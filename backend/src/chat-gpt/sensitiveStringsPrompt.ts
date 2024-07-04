export const sensitiveStringsPrompt = `
### Task
I want you to detect hardcoded sensitive information within string literals in the Java Source Code. 
Sensitive strings are those that if exposed could lead to a vulnerability or make the system vulnerable.

### Sensitive Strings Categories (suggestions):
This is not an exhaustive list. It is to give you an idea of the types of sensitive strings, 
please consider the context in which they are used. 

1) Authentication and Authorization and Credentials Information: Strings related to passwords, API keys, usernames, verification codes, etc.
2) Personal Identifiable Information (PII): Strings related to names, emails, addresses, social security numbers, health Information, etc.
3) Financial Information: Strings related to  credit cards, bank account numbers, account IDs, payment IDs, CVV, etc
4) Files Containing Sensitive Information, Sensitive File Paths, URLs: Strings related to internal URLs or file paths to files that contain sensitive information or files themselves.
5) Sensitive System and Configuration Information: Strings with database or network connection strings, database schemas, configuration details, environment variables, sensitive settings, etc.
6) Security and Encryption Information: Strings related to encryption keys, seeds, or certificates.
7) Application-Specific Sensitive Data: Strings related to sensitive information such as device IDs, Application-specific IDs, email messages, notifications, exceptions, etc.
8) Query Parameters: Strings related to sensitive data in HTTP GET requests.

### File Markers
Each file begins with "-----BEGIN FILE: [FileName]-----" and ends with "-----END FILE: [FileName]-----".

### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1) Where it says "stringName1" and "stringDescription1" you should replace with the actual name and description of the sensitive string.
2) Provide a JSON response for each file that matches the format below. 
  A) The "name" field should be the sensitive information found in the string.
  B) The "description" field should which category the string belongs to.
3) Ensure that your response doesn't break the JSON format.
{
  "files": [
    {
      "fileName": "FileName1.java",
      "sensitiveStrings": [
        {
          "name": "stringName1",
          "description": "stringDescription1"
        },
        {
          "name": "stringName2",
          "description": "stringDescription2"
        }
      ]
    }
  ]
}`
