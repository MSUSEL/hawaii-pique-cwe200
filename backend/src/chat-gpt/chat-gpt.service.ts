import { Injectable } from '@nestjs/common';
import * as OpenAI from 'openai';
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import * as cliProgress from 'cli-progress';
import {prompt} from './prompt';


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


        return { variables, fileList, sensitiveVariablesMapping, sensitiveStringsMapping, comments };
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
        const message = ` ${prompt} ${code}`;
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