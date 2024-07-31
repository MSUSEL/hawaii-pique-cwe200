export const sensitiveStringsPrompt = `
### Task
I want you to detect hardcoded sensitive information within string literals in the Java Source Code. Sensitive strings are those that, if exposed, could lead to a vulnerability or make the system vulnerable.

### Examples of Sensitive Strings
These examples are not exhaustive; many other instances may still be considered sensitive.

- String password = "secretPassword123";
- String apiKey = "ABCD-1234-EFGH-5678";
- String ssn = "123-45-6789";
- String creditCardNumber = "4111 1111 1111 1111";
- String dbConnection = "jdbc:mysql://user:password@localhost:3306/mydatabase";
- String encryptionKey = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA";

### Examples of Non-Sensitive Strings
These examples are not exhaustive; many other instances may still be considered non-sensitive.

- String welcomeMessage = "Welcome to our application!";
- String filePath = "/home/user/documents";
- String url = "http://example.com";
- String placeholderText = "Enter your name";
- String status = "Logged in successfully";
- String errorMessage = "Invalid username or password";
- String userName = "username";
- String secret = "passphrase";
- String password = "password";
- alert('hello');

### Guidelines to Reduce False Positives
1. **Context Matters:** Ensure that the string's context within the code indicates its sensitive nature. For example, "password" in a comment may not be sensitive, but in an assignment like String password = "example", it is.
2. **Common Words and Phrases:** Exclude common words and phrases unless they are in contexts indicating sensitivity (e.g., "admin" in username = "admin"; could be sensitive, but System.out.println("admin"); is not).
3. **Generic File Paths and URLs:** Exclude generic file paths and URLs that are unlikely to be sensitive, such as /home/user or http://example.com. Only include them if you are positive that they are sensitive as this contributes to many false positives.
4. **Review Surrounding Code:** Consider the surrounding code to determine if the string is part of a sensitive operation (e.g., database connection setup, user authentication process).
5. **Testing Words:** If a string has words like foo, bar, test, example, etc, it probably isn’t sensitive so don’t include them unless you are positive. 
6. **Confidence:** The main point is that you should be positive without a doubt that a string is sensitive to include it in your response. 

### File Markers
Each file begins with "-----BEGIN FILE: [FileName]-----" and ends with "-----END FILE: [FileName]-----".

### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1. Replace "stringName1" and "stringDescription1" with the actual name and description of the sensitive string.
2. Provide a JSON response for each file that matches the format below.
    - The "name" field should be the sensitive information found in the string.
    - The "description" field should indicate which category the string belongs to.
3. Do not include any non-printable or special characters in the name or description fields as this will cause the JSON to be invalid.
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
