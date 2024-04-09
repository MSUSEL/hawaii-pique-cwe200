import { Injectable } from '@nestjs/common';
import * as OpenAI from 'openai';
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import * as cliProgress from 'cli-progress';


@Injectable()
export class ChatGptService {
    openai: OpenAI.OpenAIApi = null;
    progressBar: any;
    debug: string;
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
        this.progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
        this.debug = this.configService.get('DEBUG');

    }

    /**
     * Process files into a standardized format to be appended to a GPT query
     *
     * @param files files to include in GPT prompt
     */
    async openAiGetSensitiveVariables(files: string[]) {
        let variables = [];
        let strings = [];
        let comments = [];
        const fileList: any[] = [];
        
        let sensitiveVariablesMapping = new Map<string, string[]>();
        let sensitiveStringsMapping = new Map<string, string[]>();
        let sensitiveCommentsMapping = new Map<string, string[]>();
        
        const concurrentCallsLimit = 5; // Maximum number of concurrent API calls
        const batches = []; // Array to hold all batch promises
        let completedBatches = 0; // Number of completed batches


        // Create batches of provided java code
        let batchSize = 5; 
        for (let i = 0; i < files.length; i += batchSize) {
            batches.push(files.slice(i, i + batchSize));
        }

        // Function to process a single batch
    const processBatch = async (batch: string[], index) => {
        let processedFiles = await this.fileUtilService.preprocessFiles(batch);

        if (typeof processedFiles === 'string') {
            try {
                const response = await this.createGptWithBackoff(processedFiles, index);
                completedBatches += 1;
                let progress = completedBatches / batches.length * 100;
                this.progressBar.update(progress);
                // console.log(`\r${completedBatches / batches.length * 100}% of files processed`);

                if (this.debug.toLowerCase() === 'true') {
                    console.log(`Results for batch ${index} \n ${response.message}`);
                }

                const json = JSON.parse(response.message);

                json.files.forEach((file: any) => {
                    // Extract the sensitive variables for each file
                    let fileName = file.fileName;
                    let sensitiveVariables = file.sensitiveVariables;
                    let sensitiveStrings = file.sensitiveStrings;
                    let sensitiveComments = file.sensitiveComments;

                    fileList.push({
                        fileName: fileName,
                        sensitiveVariables: sensitiveVariables,
                        sensitiveStrings: sensitiveStrings,
                        sensitiveComments: sensitiveComments
                    });

                    this.eventsGateway.emitDataToClients(
                        'data',
                        fileName + ':',
                    );
                    this.eventsGateway.emitDataToClients(
                        'data',
                        sensitiveVariables,
                    );

                    const fileVariablesList = this.extractVariableNamesMultiple(sensitiveVariables);


                    let fileStringList = this.extractVariableNamesMultiple(sensitiveStrings);  
                        
                        // fileStringList = processStrings(fileStringList);
                    
                    let fileCommentsList = this.extractVariableNamesMultiple(file.sensitiveComments);

                    variables = variables.concat(fileVariablesList);
                    sensitiveVariablesMapping[fileName] = fileVariablesList;

                    strings = strings.concat(fileStringList);
                    sensitiveStringsMapping[fileName] = fileStringList;

                    comments = comments.concat(fileCommentsList);
                    sensitiveCommentsMapping[fileName] = fileCommentsList;

                    
                });

            } catch (error) {
                console.error('Error processing GPT response:', error);
            }
        }
    };

        // Function to limit concurrent batch processing
        const limitConcurrentBatches = async (batches: string[][]) => {
            console.log("Finding Sensitive Information in Project")
            this.progressBar.start(100, 0);
            const promises = batches.map(async (batch, index) => {
                return processBatch(batch, index);
            });
            await Promise.allSettled(promises);
            this.progressBar.stop();
        };

        await limitConcurrentBatches(batches);
    
        // Post-processing
        variables = [...new Set(variables)];


        return { variables, fileList, sensitiveVariablesMapping, sensitiveStringsMapping, sensitiveCommentsMapping };
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
    async createGptWithBackoff(fileContents: string, index, retries = 10, delayMs = 1000) {
        for (let i = 0; i < retries; i++) {
            try {
                // Attempt to make a new GPT and get response
                return await this.createGpt(fileContents);
            } catch (error) {
                // Report failure
                // console.error(`Attempt ${i + 1}: Error caught in createGptWithBackoff`);

                // Calculate time until next request
                const isRateLimitError = error.response && error.response.status === 429 || error.response.status === 502;
                if (isRateLimitError && i < retries - 1) {
                    // Instead of exponential backoff, use the time specified in the header
                    try{
                        let timeOut = parseFloat(error.response.headers['x-ratelimit-reset-tokens'].replace('s', ''));
                        
                        if (this.debug.toLowerCase() === 'true'){
                            console.log(`Rate limit hit. Retrying batch ${index} in ${timeOut} seconds`)
                        }

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
        const message = `
        You are a security analyst tasked with identifying sensitive variables related to system configurations, database connections, and credentials in multiple source code files. Your goal is to identify variables that, if exposed to external users or hackers, could lead to security vulnerabilities or breaches. Please analyze the provided source code files and list down any sensitive variables related to system configurations, database connections, or credentials that fit the criteria mentioned above for each file separately. The beginning of each file is marked by "-----BEGIN FILE: [FileName]-----", and the end is marked by "-----END FILE: [FileName]-----". Please provide the names of the sensitive variables only, without disclosing any specific values, and format your response in JSON. Your analysis will help in securing the application and preventing potential data leaks or unauthorized access. I only want the JSON response not anything else. Also, give me all the sensative variables for a specifc file before moving to the next file. 
        
        Also, identify all sensitive hardcoded strings. For clarity, a 'sensitive string' is defined as any hardcoded text that could potentially contain sensitive information. This includes but is not limited to passwords, API keys, and personal information that is explicitly written in the code. Please focus on Personally Identifiable Information. Where if someone had access to the source code, they could see the information.

        Since I want your response to be in JSON and I will be taking your response and making a string array with it, I need you to make sure that it won’t break the format. Here are some formatting considerations I need you to not do.

        1) I need each sensitive string as its own element in the array. Even if there are multiple in a single concatenated string. Each of them should be by themselves. Don’t ever have + concatenated strings since this will break formatting.
        
        2) Next, don’t response with sensitive strings that will break a string array. Such as ones that have \ or nested quotes that would cause it to not be a valid string in TypeScript. IF IT HAS A backslash in it "\" DON"T SEND IT as that breaks formatting SUPER IMPORTANT. The same goes for a generic error message. 

        3) If there is a case where a string is in a key value format I would like you to just give me the value and drop the key. For example, if a result is 'Email: john.doe@example.com', only 'john.doe@example.com' should be returned. If there are multiple key value pairs in a single sensitive string, I want each of the values to be their own element. For example, "name": "Name: John Doe, Email: john.doe@example.com, Phone: 555-0100" would result in “John Doe”, “john.doe@example.com”, “555-0100”. Notice how all of the keys are dropped.   

        Just remember that it is just as important to find sensitive hardcoded strings as it is to make sure that your response does not break either JSON or String formatting. 

        
        In addition, I would like you to also provide me any sensitive information that is exposed in commments. This could be anything that is written in a comment that could be sensitive. This could be anything from a password to a personal email address.


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
            for (const variable of text) {
                    let v : string = variable.name.replace(/["\\]/g, "")
                    variables.push(`\"${v}\"`)
            }
            // variables = text.map((variable) => `\"${variable.name}\"`);
        } catch (e) {
            if (this.debug.toLowerCase() === 'true') {
                console.log(text);
            }
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
            if (this.debug.toLowerCase() === 'true') {
                console.log(text);
            }
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