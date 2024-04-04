import { Injectable } from '@nestjs/common';
import * as OpenAI from 'openai';
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';

@Injectable()
export class ChatGptService {
    openai: OpenAI.OpenAIApi = null;
    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
    ) {
        const api_key = this.configService.get('API_KEY');
        const configuration = new OpenAI.Configuration({
            apiKey: api_key,
        });
        this.openai = new OpenAI.OpenAIApi(configuration);
    }

    /**
     * Process files into a standardized format to be appended to a GPT query
     *
     * @param files files to include in GPT prompt
     */
    async openAiGetSensitiveVariables(files: string[]) {
        let variables = [];
        let strings = [];
        const fileList: any[] = [];
        let sensitiveVariablesMapping = new Map<string, string[]>();
        let sensitiveStringsMapping = new Map<string, string[]>();


        // Create batches of provided java code
        let batchSize = 5; 
        for (let i = 0; i < files.length; i += batchSize) {
            const batch = files.slice(i, i + batchSize);

            try {
                let processedFiles = await this.fileUtilService.preprocessFiles(
                    batch,
                );

                // todo is there a case when the files aren't a string?
                if (typeof processedFiles === 'string') {
                    try {
                        // Attempt to query GPT with files
                        const response = await this.createGptWithBackoff(
                            processedFiles,
                        );
                        console.log(response.message);

                        // Assuming response.message is a JSON string, parse response
                        const json = JSON.parse(response.message);
                        json.files.forEach((file: any) => {
                            // Extract the sensitive variables for each file
                            let fileName = file.fileName;
                            let sensitiveVariables = file.sensitiveVariables;
                            let sensitiveStrings = file.sensitiveStrings;

                            fileList.push({
                                key: fileName,
                                value: sensitiveVariables,
                            });

                            this.eventsGateway.emitDataToClients(
                                'data',
                                fileName + ':',
                            );
                            this.eventsGateway.emitDataToClients(
                                'data',
                                sensitiveVariables,
                            );

                            const fileVariablesList =
                                this.extractVariableNamesMultiple(
                                    sensitiveVariables,
                                );


                                let fileStringList =
                                this.extractVariableNamesMultiple(
                                    sensitiveStrings,
                                );  
                                
                                // fileStringList = processStrings(fileStringList);

                            variables = variables.concat(fileVariablesList);
                            sensitiveVariablesMapping[fileName] = fileVariablesList;

                            strings = strings.concat(fileStringList);
                            sensitiveStringsMapping[fileName] = fileStringList;
                            
                        });
                    } catch (error) {
                        console.error('Error processing GPT response:', error);
                        // Handle the error or continue
                    }
                }
            } catch (error) {
                console.error('Error in file processing:', error);
                // Handle the error or continue
            }
        }
        // Ensure unique variables
        variables = [...new Set(variables)];
        return { variables, fileList, sensitiveVariablesMapping, sensitiveStringsMapping };
    }

    async createDavinci(fileContents: string) {
        var prompt = this.createQuery(fileContents);
        var response = await this.createDavinciCompletion(prompt);
        this.extractVariableNames(response.message);
        return response;
    }


    /**
     * Create new GPT with given prompt and get response
     *
     * @param fileContents content to make prompt with
     */
    async createGpt(fileContents: string) {
        const prompt = this.createQuery(fileContents);
        // this.extractVariableNames(response.message);
        return await this.createGptFourCompletion(prompt);
    }

    /**
     * @param milliseconds time to delay in milliseconds
     */
    async delay(milliseconds) {
        return new Promise((resolve) => setTimeout(resolve, milliseconds));
    }

    /**
     * Create GPT with backoff parameters
     *
     * @param fileContents contents to create GPT with
     * @param retries number of request tries
     * @param delayMs delay between requests in milliseconds
     */
    async createGptWithBackoff(fileContents: string, retries = 10, delayMs = 1000) {
        for (let i = 0; i < retries; i++) {
            try {
                // Attempt to make a new GPT and get response
                return await this.createGpt(fileContents);
            } catch (error) {
                // Report failure
                console.error(
                    `Attempt ${i + 1}: Error caught in createGptWithBackoff`,
                );

                // Calculate time until next request
                const isRateLimitError = error.response && error.response.status === 429;
                if (isRateLimitError && i < retries - 1) {
                    // Instead of exponential backoff, use the time specified in the header
                    try{
                        let timeOut = parseFloat(error.response.headers['x-ratelimit-reset-tokens'].replace('s', ''));
                        console.log(`Rate limit hit. Retrying in ${timeOut} seconds`)
                        await this.delay(timeOut * 1000);
                    }
                    // If there is an issue with the header, use exponential backoff
                    catch(e){
                        await this.delay(delayMs * Math.pow(2, i)); // Exponential backoff
                    }
                  
                } else {
                    throw error; // Re-throw the error if it's not a 429 or if max retries exceeded
                }
            }
        }
        throw new Error('createGptWithBackoff: Max retries exceeded');
    }

    /**
     * Create new GPT query
     *
     * @param code code to append to query prompt
     */
    createQuery(code: string) {
        // `You are a security analyst tasked with identifying sensitive variables related to system configurations, database connections, and credentials, which could potentially have security issues in a given source code. Your goal is to identify variables that, if exposed to external users or hackers, could lead to security vulnerabilities or breaches. Please analyze the provided source code and list down any sensitive variables related to system configurations, database connections, or credentials that fit the criteria mentioned above. Please provide the names of the sensitive variables only, without disclosing any specific values. Your analysis will help in securing the application and preventing potential data leaks or unauthorized access.please I want your response in json format like
        // {
        //     "sensitiveVariables": [
        //       {
        //         "name": "variableName1",
        //         "description": "variableDescription"
        //       },
        //       {
        //         "name": "variableName2",
        //         "description": "variableDescription"
        //       },
        //       {
        //         "name": "variableNameN",
        //         "description": "variableDescription"
        //       }
        //     ]
        // }

        // New Prompt
        // todo move to a const config?
        const message = `
        You are a security analyst tasked with identifying sensitive variables related to system configurations, database connections, and credentials in multiple source code files. Your goal is to identify variables that, if exposed to external users or hackers, could lead to security vulnerabilities or breaches. Please analyze the provided source code files and list down any sensitive variables related to system configurations, database connections, or credentials that fit the criteria mentioned above for each file separately. The beginning of each file is marked by "-----BEGIN FILE: [FileName]-----", and the end is marked by "-----END FILE: [FileName]-----". Please provide the names of the sensitive variables only, without disclosing any specific values, and format your response in JSON. Your analysis will help in securing the application and preventing potential data leaks or unauthorized access. I only want the JSON response not anything else. Also, give me all the sensative variables for a specifc file before moving to the next file. 
        
        Also, identify all sensitive hardcoded strings. For clarity, a 'sensitive string' is defined as any hardcoded text that could potentially contain sensitive information. This includes but is not limited to passwords, API keys, and personal information that is explicitly written in the code. Where if someone had access to the source code, they could see the information.

        Since I want your response to be in JSON and I will be taking your response and making a string array with it, I need you to make sure that it won’t break the format. Here are some formatting considerations I need you to not do.

        1) I need each sensitive string as its own element in the array. Even if there are multiple in a single concatenated string. Each of them should be by themselves. Don’t ever have + concatenated strings since this will break formatting.
        
        2) Next, don’t response with sensitive strings that will break a string array. Such as ones that have \ or nested quotes that would cause it to not be a valid string in TypeScript. IF IT HAS A \ DON"T SEND IT. The same goes for a generic error message. 

        3) If there is a case where a string is in a key value format I would like you to just give me the value and drop the key. For example, if a result is 'Email: john.doe@example.com', only 'john.doe@example.com' should be returned. If there are multiple key value pairs in a single sensitive string, I want each of the values to be their own element. For example, "name": "Name: John Doe, Email: john.doe@example.com, Phone: 555-0100" would result in “John Doe”, “john.doe@example.com”, “555-0100”. Notice how all of the keys are dropped.   

        Just remember that it is just as important to find sensitive hardcoded strings as it is to make sure that your response does not break either JSON or String formatting. 

        


        
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
                ]
              }
            ]
          }
          

        ${code}`;
        return message;
    }

    async createDavinciCompletion(prompt: string) {
        const completion = await this.openai.createCompletion({
            model: 'text-davinci-003',
            prompt: prompt,
            temperature: 0.1,
            max_tokens: 250,
        });
        return { message: completion?.data.choices?.[0]?.text };
    }

    async createGptCompletion(prompt: string) {
        var completion = await this.openai.createChatCompletion({
            model: 'gpt-3.5-turbo',
            temperature: 0.2,
            messages: [{ role: 'user', content: prompt }],
        });
        return { message: completion.data.choices[0].message.content };
    }


    /**
     * Execute GPT request
     *
     * @param prompt GPT prompt
     */
    async createGptFourCompletion(prompt: string) {
        try {
            const completion = await this.openai.createChatCompletion({
                model: 'gpt-4',
                temperature: 0.2,
                messages: [{ role: 'user', content: prompt }],
            });

            return { message: completion.data.choices[0].message.content };
        } catch (error) {
            throw error;
        }
    }

    extractVariableNamesMultiple(text: SensitiveVariables[]): string[] {
        var variables = [];
        try {
            variables = text.map((variable) => `\"${variable.name}\"`);
            return variables;
        } catch (e) {
            console.log(text);
            return variables;
        }
        return variables;
    }

    extractVariableNames(text: string): string[] {
        var variables = [];
        try {
            var json = JSON.parse(text);
            variables = json['sensitiveVariables'].map(
                (variable) => `\"${variable.name}\"`,
            );
            return variables;
        } catch (e) {
            console.log(text);
            return variables;
        }
    }

    async getFileGptResponse(filePath: String) {
        var directories = filePath.split('\\');
        var jsonFilePath = path.join(
            directories[0],
            directories[1],
            'data.json',
        );
        try {
            var jsonArray = await this.fileUtilService.readJsonFile(jsonFilePath);
            for (const obj of jsonArray) {
                if (obj.key === filePath) {
                    return obj;
                }
            }
        } catch (e) {
            return {
                value: 'no data found for this file yet. please re run your project to get the results',
            };
        }
    }
}

interface SensitiveVariables {
    name: string;
    description: string;
}

function processStrings(strings: string[]): string[] {
    return strings
        .map((str) => {
            // Attempt to split the string into key and value by ":"
            const parts = str.split(/:\s*/);
            // If there's a key-value structure, keep the value part
            if (parts.length > 1) {
                str = parts[1];
            }
            return str;
        })
        .filter((str) => {
            // Drop the string if it contains backslashes or unescaped double quotes
            return !str.includes('\\') && !str.match(/(?<!\\)"/);
        });
}