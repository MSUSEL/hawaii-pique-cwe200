export const sensitiveVariablesPrompt = `
### Task
I want you to detect sensitive variables in the Java Source Code. Sensitive variables are those that if 
exposed could lead to a vulnerability or make the system vulnerable.

### Sensitive Variable Categories (suggestions):
This is not an exhaustive list. It is to give you an idea of the types of sensitive variables, 
please consider the context in which they are used. 

1) Authentication and Authorization and Credentials Information: Variables holding passwords, API keys, usernames, verification codes, etc.
2) Personal Identifiable Information (PII): Variables containing names, emails, addresses, social security numbers, health Information, etc.
3) Financial Information: Variables related to credit cards, bank account numbers, account IDs, payment IDs, CVV, etc
4) Files Containing Sensitive Information, Sensitive File Paths, URLs: Variables storing internal URLs or file paths to files that contain sensitive information or files themselves.
5) Sensitive System and Configuration Information: Variables with database or network connection strings, database schemas, configuration details, environment variables, sensitive settings, etc.
6) Security and Encryption Information: Variables holding encryption keys, seeds, or certificates.
7) Application-Specific Sensitive Data: Variables storing sensitive information such as device IDs, Application-specific IDs, email messages, notifications, exceptions, etc.
8) Query Parameters: Variables storing sensitive data in HTTP GET requests.

### File Markers
Each file begins with "-----BEGIN FILE: [FileName]-----" and ends with "-----END FILE: [FileName]-----".

### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1) Where it says "variableName1" and "variableDescription1" you should replace them with the actual name and description of the sensitive variable.
2) Provide a JSON response for each file that matches the format below. 
  A) The "name" field should be the sensitive information found in the variable.
  B) The "description" field should tell which category the variable belongs to.
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
}
`
