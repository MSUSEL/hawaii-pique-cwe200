export const variablesPrompt = `
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

Please respond in this format 
{
  "files": [
    {
      "fileName": "FileName1.java",
      "variables": [
        {
          "name": "variableName",
          "isSensitive": "yes/no",
          "reason": "reason for the decision"
        }
      ]
    }
  ]
}`

