import { Injectable } from '@nestjs/common';
import * as OpenAI from 'openai';
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import * as cliProgress from 'cli-progress';
import {prompt} from './prompt';
import { response } from 'express';
import {encode, decode} from 'gpt-tokenizer';
import async from 'async';

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
        
        let completedFiles = 0; // Number of completed files

        const res = await this.dynamicBatching(files);
        const batches = res.batchesOfText;
        const filesPerBatch = res.filesPerBatch;


        // Function to process a single batch
    const processBatch = async (batch: string, filesInBatch : number, index : number) => {
            try {
                const response = await this.createGptWithBackoff(batch, index);                
                completedFiles += filesInBatch;
                this.progressBar.update(completedFiles);

                if (this.debug.toLowerCase() === 'true') {
                    console.log(`Results for batch ${index} \n ${response.message}`);
                }
                
                let json = extractAndParseJSON(response.message)
                
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

                    this.eventsGateway.emitDataToClients('data',fileName + ':',);
                    this.eventsGateway.emitDataToClients('data',sensitiveVariables,);
                    
                    // Extract the sensitive variables, strings, and comments for each file
                    const fileVariablesList = this.extractVariableNamesMultiple(sensitiveVariables);
                    let fileStringList = this.extractVariableNamesMultiple(sensitiveStrings);  
                    let fileCommentsList = this.extractVariableNamesMultiple(file.sensitiveComments);

                    // Create Mappings for the YML files
                    // Append or create new entry in sensitiveVariablesMapping
                    if (sensitiveVariablesMapping[fileName]) {
                        sensitiveVariablesMapping[fileName] = sensitiveVariablesMapping[fileName].concat(fileVariablesList);
                    } else {
                        sensitiveVariablesMapping[fileName] = fileVariablesList;
                    }

                    // Append or create new entry in sensitiveStringsMapping
                    if (sensitiveStringsMapping[fileName]) {
                        sensitiveStringsMapping[fileName] = sensitiveStringsMapping[fileName].concat(fileStringList);
                    } else {
                        sensitiveStringsMapping[fileName] = fileStringList;
                    }

                    // Append or create new entry in sensitiveCommentsMapping
                   comments = comments.concat(fileCommentsList);
                   sensitiveCommentsMapping[fileName] = fileCommentsList;
                });

            } catch (error) {
                console.error('Error processing GPT response:', error);
            }
    };

        // Function to limit concurrent batch processing
        const processConcurrentBatches = async (batches, filesPerBatch ) => {
            let concurrencyLimit = 50; // Number of concurrent tasks to run
            console.log("Finding Sensitive Information in Project");
        
            const totalFiles = filesPerBatch.reduce((acc, num) => acc + num, 0);
            this.progressBar.start(totalFiles, 0);
        
            const queue = async.queue(async (task, callback) => {
                await processBatch(task.batch, task.files, task.index);
                callback();
            }, concurrencyLimit);
        
            // Push tasks to the queue
            batches.forEach((batch, index) => {
                queue.push({ batch, files: filesPerBatch[index], index });
            });
        
            // Wait for all tasks to be processed
            await queue.drain();
            this.progressBar.stop();
        };

        await processConcurrentBatches(batches, filesPerBatch);
    
        // Post-processing
        variables = [...new Set(variables)];


        return { variables, fileList, sensitiveVariablesMapping, sensitiveStringsMapping, comments };
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
    async createGptWithBackoff(fileContents: string, index, retries = 100, delayMs = 1000) {
        for (let i = 0; i < retries; i++) {
            try {
                // Attempt to make a new GPT and get response
                return await this.createGptFourCompletion(fileContents);
            } catch (error) {
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
     * Execute GPT request
     *
     * @param prompt GPT prompt
     */
    async createGptFourCompletion(prompt: string) {
        try {
            const completion = await this.openai.createChatCompletion({
                model: 'gpt-4o',
                temperature: 0.2,
                messages: [{ role: 'user', content: prompt }],
                top_p: 0.8,
                frequency_penalty: 0.1,
                presence_penalty: 0.1,
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


    async dynamicBatching(files) {
        const maxTokensPerBatch = 10000; // Maximum number of tokens per batch
        const promptTokenCount = encode(prompt).length;
        let batchesOfText = []; // Array to hold all batches of text
        let filesPerBatch = []; // Used later on by the progress bar
        let batchText = prompt; 
        let totalBatchTokenCount = promptTokenCount; 
        let currentFileCount = 0;
        let i = 0;
    
        for (const file of files) {
            // Clean up the file (Removes unnecessary whitespace)
            const fileContent = await this.fileUtilService.processJavaFile(file);
    
            // Add the start and end bounds to the file and then get its token count
            const fileTokens = encode(await this.fileUtilService.addFileBoundaryMarkers(file, fileContent));
            const currentFileTokenCount = fileTokens.length;
    
            if (currentFileTokenCount > maxTokensPerBatch) {
                console.log(`File ${file} is too large to fit in a single batch ${i}`);
    
                // If there is already content in the current batch, push it and start a new batch
                if (totalBatchTokenCount > promptTokenCount) {
                    batchesOfText.push(batchText);
                    filesPerBatch.push(currentFileCount);
                    i = 0;
                    batchText = prompt; // Start with a fresh batch that includes the prompt
                    currentFileCount = 0; // Reset file count for the new batch
                }
    
                let startIndex = 0;
                let endIndex = 0;
    
                while (startIndex < fileContent.length) {
                    endIndex = startIndex + Math.min(maxTokensPerBatch * 3, fileContent.length - startIndex);
    
                    const sliceWithBounds = await this.fileUtilService.addFileBoundaryMarkers(file, fileContent.slice(startIndex, endIndex));
                    let totalsize = fileContent.length
                    let slicesize = sliceWithBounds.length
    
                    // Create a new batch with the prompt and the current slice
                    batchText = prompt + sliceWithBounds;
                    let sliceTokens = encode(batchText).length;
    
                    batchesOfText.push(batchText);  // Push the new batch immediately
                    filesPerBatch.push(1); // Each slice from an oversized file is treated as a separate batch with 1 file
                    i = 0;
                    // Update the startIndex for the next slice
                    startIndex = endIndex;
                }
            }
            // Current batch full, push it and start a new one
            else if (totalBatchTokenCount + currentFileTokenCount > maxTokensPerBatch) {
                batchesOfText.push(batchText);
                filesPerBatch.push(currentFileCount);
                i = 0;
                batchText = prompt + await this.fileUtilService.addFileBoundaryMarkers(file, fileContent);
                totalBatchTokenCount = promptTokenCount + currentFileTokenCount;
                currentFileCount = 1; // Start counting files in the new batch
            } 
            
            // The current file can be added to the current batch
            else {
                batchText += await this.fileUtilService.addFileBoundaryMarkers(file, fileContent);
                totalBatchTokenCount += currentFileTokenCount;
                currentFileCount++; // Increment the count of files in the current batch
            }
            i += 1;
        }
    
        // Add the last batch if it contains any content beyond the initial prompt
        if (batchText.length > prompt.length) {
            batchesOfText.push(batchText);
            filesPerBatch.push(currentFileCount);
        }
    
        return { batchesOfText, filesPerBatch };
    }

    // With GPT-4o this approach might work the best
    async simpleBatching(files){
        let batchesOfText = []
        let filesPerBatch = []
        for (const file of files){
            const fileContent = await this.fileUtilService.addFileBoundaryMarkers(file, await this.fileUtilService.processJavaFile(file))
            batchesOfText.push(prompt + fileContent)
            filesPerBatch.push(1)
        }
        return {batchesOfText, filesPerBatch}
    }
    
}

interface SensitiveVariables {
    name: string;
    description: string;
}

function extractAndParseJSON(inputString) {
    // Attempt to sanitize input by escaping problematic characters
    const sanitizedInput = inputString.replace(/(\+)(\s*[\w\s]+)(\+)/g, (match, p1, p2, p3) => `"${p1}${p2.trim()}${p3}"`);

    // Regular expression to find JSON objects or arrays
    const jsonRegex = /{.*}|\[.*\]/s;

    // Match against the sanitized input string
    const match = sanitizedInput.match(jsonRegex);

    if (match) {
        try {
            // Try parsing the JSON string found
            const json = JSON.parse(match[0]);
            return json;
        } catch (error) {
            // Log or handle JSON parsing errors
            console.error("Failed to parse JSON:", error);
            return null;
        }
    } else {
        // No JSON found
        console.log("No JSON found in the input string.");
        return null;
    }
}

