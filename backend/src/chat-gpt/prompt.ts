export const prompt = `
       You are a security analyst tasked with identifying sensitive variables related to system configurations, database connections, and credentials in multiple source code files. Your goal is to identify variables that, if exposed to external users or hackers, could lead to security vulnerabilities or breaches. Please analyze the provided source code files and list down any sensitive variables related to system configurations, database connections, or credentials that fit the criteria mentioned above for each file separately. The beginning of each file is marked by "-----BEGIN FILE: [FileName]-----", and the end is marked by "-----END FILE: [FileName]-----". Please provide the names of the sensitive variables only, without disclosing any specific values, and format your response in JSON. Your analysis will help in securing the application and preventing potential data leaks or unauthorized access. I only want the JSON response not anything else. Also, give me all the sensative variables for a specifc file before moving to the next file. 
        
        Also, identify all sensitive hardcoded string literals in the code. For clarity, a 'sensitive string' is defined as any hardcoded string literal text that contains sensitive information. This includes but is not limited to passwords, API keys, and personal information that is explicitly written in the code. Where if someone had access to the source code, they could see the information.
        I need your response to make sure that it won’t break the format. Here are some formatting considerations.
        1) I need each sensitive string as its own element in the array. Even if there are multiple in a single concatenated string. Each of them should be by themselves. Don’t ever have + concatenated strings since this will break formatting.
        2) If there is a case where a string is in a key value format I would like you to just give me the value and drop the key. For example, if a result is 'Email: john.doe@example.com', only 'john.doe@example.com' should be returned. If there are multiple key value pairs in a single sensitive string, I want each of the values to be their own element. For example, "name": "Name: John Doe, Email: john.doe@example.com, Phone: 555-0100" would result in “John Doe”, “john.doe@example.com”, “555-0100”. Notice how all of the keys are dropped.   
        Just remember that it is just as important to find sensitive hardcoded strings as it is to make sure that your response does not break either JSON or String formatting. 

        In addition, I would like you to also provide me any sensitive information that is exposed in commments. In particular, I am looking for hardcoded sensitive information. 
        What I don't want is any comments that are just generic comments that don't have any sensitive information in them. 

        Please structure your response in the following JSON format for each file:

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
                ],
                "sensitiveStrings": [
                  {
                    "name": "stringName1",
                    "description": "stringDescription1"
                  },
                  {
                    "name": "stringName2",
                    "description": "stringDescription2"
                  }
                ],
                sensitiveComments: [
                    {
                        "name": "commentName1",
                        "description": "commentDescription1"
                    },
                    {
                        "name": "commentName2",
                        "description": "commentDescription2"
                    }
                ]
              },
              {
                "fileName": "FileName2.java",
                "sensitiveVariables": [
                  {
                    "name": "variableName1",
                    "description": "variableDescription1"
                  },
                  {
                    "name": "variableName2",
                    "description": "variableDescription2"
                  }
                ],
                "sensitiveStrings": [
                  {
                    "name": "stringName1",
                    "description": "stringDescription1"
                  },
                  {
                    "name": "stringName2",
                    "description": "stringDescription2"
                  }
                ],
                sensitiveComments: [
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


        //   You are a security analyst assigned to identify potential sources of CWE-200 vulnerabilities in multiple source code files. Your task is to scrutinize the provided source code files for sensitive variables, hardcoded strings, and comments that could, if exposed, lead to information disclosure vulnerabilities. Each file will be clearly marked with "-----BEGIN FILE: [FileName]-----" and "-----END FILE: [FileName]-----". Your detailed objectives are:

        //   Sensitive Variables Identification:
        //   Identify variables specifically tied to system configurations, database connections, or authentication credentials, which are common sources of information exposure.
        //   List each sensitive variable by name, ensuring no actual values are disclosed, and focus solely on the variable names that could be vulnerable if exposed.
          
        //   Hardcoded Sensitive Strings:
        //   Identify any hardcoded strings within the code that contain potentially sensitive information such as passwords, API keys, or personal identifiers. These are crucial as they directly contribute to information exposure vulnerabilities.
        //   Formatting Key-Value Pairs: If a sensitive string is formatted as a key-value pair, extract and return only the values, omitting the keys. For instance, if a string is 'Email: john.doe@example.com', return only 'john.doe@example.com'. If the string contains multiple key-value pairs such as "name": "Name: John Doe, Email: john.doe@example.com, Phone: 555-0100", extract and list each value separately: ["John Doe", "john.doe@example.com", "555-0100"]. This ensures that the keys, which do not contain sensitive information by themselves, do not clutter the response or break the formatting.
        //   Each sensitive string should be returned as its own distinct element in the output array, avoiding concatenation or merging of separate strings, as this maintains clarity and prevents format errors in the JSON response.
          
        //   Sensitive Comments:
        //   Extract and report any comments containing hardcoded sensitive information or hints about system configurations that could aid an attacker in gaining unauthorized access or information.
          
