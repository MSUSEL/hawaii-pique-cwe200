export const sensitiveVariablesPrompt = `
### Task
I want you to detect sensitive variables in Java Source Code. 
Sensitive variables are those that, if exposed, could lead to a vulnerability or compromise the security and integrity of a system.

### Sensitive Variable Categories 

1) Authentication and Authorization and Credentials Information: Variables holding passwords, API keys, keys, usernames, verification codes, credentials etc.
2) Personal Identifiable Information (PII): Variables containing names, emails, addresses, social security numbers, health information, accounts etc.
3) Financial Information: Variables related to credit cards, bank account numbers, account IDs, payment IDs, CVV, etc
4) Files Containing Sensitive Information, Sensitive File Paths, URLs/URIs: Variables storing internal URLs/URIs or file paths to files that contain sensitive information or files themselves.
5) Sensitive System and Configuration Information: Variables with database, cloud provider, or network connection strings, database schemas, configuration details, environment variables, sensitive settings, controllers, and managers.
6) Security and Encryption Information: Variables holding encryption keys, seeds, or certificates.
7) Application-Specific Sensitive Data: Variables storing sensitive information such as device details (Names, IDs, properties, objects), Application-specific IDs, email messages, notifications, etc.
8) Query Parameters: Variables storing sensitive data in HTTP GET requests.
9) Exceptions: Variables storing exceptions(e), exception messages, error messages, etc.

### Note
Exclude variables related to handlers, wrappers, loggers, listeners, generic file paths, and URLs.

### File Markers
Each file begins with "-----BEGIN FILE: [FileName]-----" and ends with "-----END FILE: [FileName]-----".

### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1) Where it says "variableName1" and "variableDescription1" you should replace them with the actual name and description of the sensitive variable.
2) Provide a JSON response for each file that matches the format below. 
  A) The "name" field should be the name of the sensitive variable.
3) Make sure there are no duplicate entries in the response.
{
  "files": [
    {
      "fileName": "FileName1.java",
      "variables": [
        {
          "name": "variableName1",
        },
        {
          "name": "variableName2",
        }
      ]
    }
  ]
}`

