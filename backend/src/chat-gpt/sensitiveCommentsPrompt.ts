export const sensitiveCommentsPrompt = `
### Task
I want you to detect sensitive information within comments in the Java Source Code. Sensitive comments are those that if 
exposed could lead to a vulnerability or make the system vulnerable.

### Sensitive Comments Categories (suggestions):
This is not an exhaustive list. It is to give you an idea of the types of sensitive comments, 
please consider the context in which they are used. 

1) Authentication and Authorization and Credentials Information: Comments related to passwords, API keys, usernames, verification codes, etc.
2) Personal Identifiable Information (PII): Comments related to names, emails, addresses, social security numbers, health Information, etc.
3) Financial Information: Comments related to  credit cards, bank account numbers, account IDs, payment IDs, CVV, etc
4) Files Containing Sensitive Information, Sensitive File Paths, URLs: Comments related to internal URLs or file paths to files that contain sensitive information or files themselves.
5) Sensitive System and Configuration Information: Comments with database or network connection strings, database schemas, configuration details, environment variables, sensitive settings, etc.
6) Security and Encryption Information: Comments related to encryption keys, seeds, or certificates.
7) Application-Specific Sensitive Data: Comments related to sensitive information such as device IDs, Application-specific IDs, email messages, notifications, exceptions, etc.
8) Query Parameters: Comments related to sensitive data in HTTP GET requests.

### File Markers
Each file begins with "-----BEGIN FILE: [FileName]-----" and ends with "-----END FILE: [FileName]-----".

### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1) Where it says "commentName1" and "commentDescription1" you should replace with the actual name and description of the sensitive comment.
2) Provide a JSON response for each file that matches the format below. 
  A) The "name" field should be the sensitive information found in the comment.
  B) The "description" field should tell which category the comment belongs to.
3) Ensure that your response doesn't break the JSON format.
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
