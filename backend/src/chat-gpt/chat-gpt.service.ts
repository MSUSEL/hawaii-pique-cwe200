import { Injectable } from '@nestjs/common';
import * as OpenAI from 'openai';
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import * as cliProgress from 'cli-progress';
import {sensitiveVariablesPrompt} from './sensitiveVariablesPrompt';
import {sensitiveStringsPrompt} from './sensitiveStringsPrompt';
import {sensitiveCommentsPrompt} from './sensitiveCommentsPrompt';
import { sinkPrompt } from './sinkPrompt';
import {classifyPrompt} from './classifyPrompt'
import { response } from 'express';
import { get_encoding } from 'tiktoken';
import async from 'async';
import { spawn } from 'child_process';
import { Ollama } from 'ollama-node';


@Injectable()
export class ChatGptService {
    openai: OpenAI.OpenAIApi = null;
    progressBar: any;
    debug: string;
    projectsPath: string;
    encode: any;
    idToNameMapping: Map<string, string> = new Map<string, string>();
    parsedResults:{ [key: string]: JavaParseResult };
    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
    ) {
        const api_key = this.configService.get('API_KEY');
        const configuration = new OpenAI.Configuration({
            apiKey: api_key,
        });

        this.projectsPath = this.configService.get<string>(
            'CODEQL_PROJECTS_DIR',
        );
        this.openai = new OpenAI.OpenAIApi(configuration);
        this.progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
        this.debug = this.configService.get('DEBUG');
        this.encode = get_encoding("o200k_base");
        this.parsedResults = {};


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
        let classifications = [];
        let sinks = [];
        const fileList: any[] = [];
        
        let sensitiveVariablesMapping = new Map<string, string[]>();
        let sensitiveStringsMapping = new Map<string, string[]>();
        let sensitiveCommentsMapping = new Map<string, string[]>();
        let classificationMapping = new Map<string, string[]>();
        let sinksMapping = new Map<string, string[][]>();
        let rawResponses = "";
        let parsedResults: { [key: string]: JavaParseResult } = {};

        
        let completedFiles = 0; // Number of completed files
    
        const prompts = [
            { type: 'variables', prompt: sensitiveVariablesPrompt, mapping: sensitiveVariablesMapping, result: variables },
            // { type: 'strings', prompt: sensitiveStringsPrompt, mapping: sensitiveStringsMapping, result: strings },
            // { type: 'comments', prompt: sensitiveCommentsPrompt, mapping: sensitiveCommentsMapping, result: comments },
            // { type: 'sinks', prompt: sinkPrompt, mapping: sinksMapping, result : sinks}

        ];
    
        // Dictionary to store results by file name
        const fileResults = {};
    
        for (const { type, prompt, mapping, result } of prompts) {
            // const res = await this.dynamicBatching(files, prompt);
            const res = await this.simpleBatching(files, prompt, type); // Use this if you just want to send one file at a time

            const batches = res.batchesOfText;
            const filesPerBatch = res.filesPerBatch;
    
            const processBatch = async (batch: string, filesInBatch: number, index: number) => {
                try {
                    const response = await this.createGptWithBackoff(batch, index);
                    rawResponses += this.replaceIDs(response.message)

                    completedFiles += filesInBatch;
                    this.progressBar.update(completedFiles);
                    // this.eventsGateway.emitDataToClients('', this.progressBar)

    
                    if (this.debug.toLowerCase() === 'true') {
                        console.log(`Results for batch ${index} \n ${response.message}`);
                    }
                    
                    let json = extractAndParseJSON(response.message);
                    
                    json.files.forEach((file: any) => {
                        let fileID = file.fileName.split('.java')[0];
                        let fileName = this.idToNameMapping.get(fileID);
                        let sensitiveData = [];
                        if (type === 'variables') {sensitiveData = file.sensitiveVariables}
                        else if (type === 'strings') {sensitiveData = file.sensitiveStrings}
                        else if (type === 'comments') {sensitiveData = file.sensitiveComments}
                        else if (type === 'classification') {sensitiveData = file.classification}
                        else if (type === 'sinks') {sensitiveData = file.sinks}

                        sensitiveData = sensitiveData.filter((value, index, self) => 
                            index === self.findIndex((t) => (
                                t.name === value.name && t.description === value.description
                            ))
                        );                        
    
                        // If file already exists in the dictionary, append the data
                        if (fileResults[fileName]) {
                            fileResults[fileName][type] = Array.from(new Set(fileResults[fileName][type].concat(sensitiveData)));                        } else {
                            
                            // Initialize the object with empty arrays if it does not exist
                            fileResults[fileName] = {
                                fileName: fileName,
                                variables: [],
                                strings: [],
                                comments: [],
                                classification: [],
                                sinks: [],
                            };
                            fileResults[fileName][type] = (sensitiveData);
                        }
    
                        // this.eventsGateway.emitDataToClients('data', fileName + ':',);
                        // this.eventsGateway.emitDataToClients('data', sensitiveData,);
    
                        let fileDataList = this.extractVariableNamesMultiple(sensitiveData);
                        
                        // If type is sinks, extract the types
                        if (type === 'sinks') {
                            const names = this.extractVariableNamesMultiple(sensitiveData);
                            const types = this.extractTypes(sensitiveData);
                          
                            let values: string[][] = names.map((name, index) => [name, types[index]]);
                          
                            if (sinksMapping.has(fileName)) {
                              sinksMapping.set(fileName, sinksMapping.get(fileName)!.concat(values));
                            } else {
                              sinksMapping.set(fileName, values);
                            }
                          }
                        else{
                            if (mapping[fileName]) {
                                mapping[fileName] = mapping[fileName].concat(fileDataList);
                            } else {
                                mapping[fileName] = fileDataList;
                            }
                        }    
                        result.push(...fileDataList);
                    });
    
                } catch (error) {
                    console.error('Error processing GPT response:', error);
                }
            };
    
            const processConcurrentBatches = async (batches, filesPerBatch) => {
                let concurrencyLimit = 50; // Number of concurrent tasks to run
                console.log(`Finding ${type} in Project`);
            
                const totalFiles = filesPerBatch.reduce((acc, num) => acc + num, 0);
                this.progressBar.start(totalFiles, 0);
            
                const queue = async.queue(async (task, callback) => {
                    await processBatch(task.batch, task.files, task.index);
                    callback();
                }, concurrencyLimit);
            
                batches.forEach((batch, index) => {
                    queue.push({ batch, files: filesPerBatch[index], index });
                });
            
                await queue.drain();
                this.progressBar.stop();
            };
            // Write raw responses to a file
            this.fileUtilService.writeToFile(path.join(this.projectsPath, 'rawResponses.txt'), rawResponses);
            await processConcurrentBatches(batches, filesPerBatch);
        }
    
        // Convert the dictionary to a list
        let numParsedFiles = 0;
        for (const fileName in fileResults) {
            fileList.push(fileResults[fileName]);
            numParsedFiles += 1;
        }
    
        variables = [...new Set(variables)];
        strings = [...new Set(strings)];
        comments = [...new Set(comments)];
        classifications = [...new Set(classifications)];
        sinks = [...new Set(sinks)];

        // A sanity check to ensure that the number of files processed is equal to the number of files in the project
        console.log(`Total number of files {${files.length}}, total number of parsed files {${numParsedFiles}}`)
    
        return { variables, strings, comments, fileList, sensitiveVariablesMapping, sensitiveStringsMapping, sensitiveCommentsMapping, classifications, classificationMapping, sinks, sinksMapping };
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
                // return await this.createLlama3Completion(fileContents);
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
            });

            return { message: completion.data.choices[0].message.content };
        } catch (error) {
            throw error;
        }
    }


    async createLlama3Completion(prompt: string) {
        try {
            const ollama = new Ollama();
            await ollama.setModel("llama3");
            
            // callback to print each word 
            const print = (word: string) => {
              process.stdout.write(word);
            }
            let response = await ollama.streamingGenerate(prompt, print);
            return { message: response };

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

    extractTypes(text: SinkType[]): string[] {
        var types = [];
        try {
            for (const variable of text) {
                    let v : string = variable.type.replace(/["\\]/g, "")
                    types.push(`\"${v}\"`)
            }
        } catch (e) {
            if (this.debug.toLowerCase() === 'true') {
                console.log(text);
            }
        }
        return types;
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


    async dynamicBatching(files, prompt) {
        const maxTokensPerBatch = 5000;
        const promptTokenCount = this.encode.encode(prompt).length;
        let batches = [];
        let currentBatch = { text: prompt, tokenCount: promptTokenCount, fileCount: 0 };
        
        for (const file of files) {
            const fileID = `ID-${this.idToNameMapping.size}`;
            this.idToNameMapping.set(fileID, file.split('\\').pop());
            const fileContent = await this.fileUtilService.processJavaFile(file, fileID);
            const boundedFileContent = this.fileUtilService.addFileBoundaryMarkers(fileID, fileContent);
            const fileTokenCount = this.encode.encode(boundedFileContent).length + promptTokenCount;
            // const variables = await this.fileUtilService.parseJavaFile(file);
            // console.log(`Here are the variables for ${file} ${variables.join(", ")}`);
    
            if (fileTokenCount > maxTokensPerBatch) {
                // Case 3: File is too large and needs to be split
                if (currentBatch.fileCount > 0) this.pushCurrentBatch(batches, currentBatch, prompt);
                this.handleLargeFile(fileID, fileContent, batches, prompt, maxTokensPerBatch, currentBatch);
                continue;
            }
    
            if (currentBatch.tokenCount + fileTokenCount > maxTokensPerBatch) {
                // Case 2: File doesn't fit in the current batch
                this.pushCurrentBatch(batches, currentBatch, prompt);
            }
    
            // Case 1: File fits in the current batch
            this.addFileToBatch(currentBatch, boundedFileContent, fileTokenCount);
        }
        
        // Push the last batch if it's not empty
        if (currentBatch.fileCount > 0) this.pushCurrentBatch(batches, currentBatch, prompt);
    
        return { batchesOfText: batches.map(b => b.text), filesPerBatch: batches.map(b => b.fileCount) };
    }
    
    addFileToBatch(batch, content, tokenCount) {
        batch.text += content;
        batch.tokenCount += tokenCount;
        batch.fileCount++;
    }
    
    pushCurrentBatch(batches, batch, prompt) {
        batches.push({ ...batch });
        batch.text = prompt; // Reset to just the prompt
        batch.tokenCount = this.encode.encode(batch.text).length;
        batch.fileCount = 0;
    }
    
    handleLargeFile(fileID, content, batches, prompt, maxTokensPerBatch, currentBatch) {
        let startIndex = 0;
        while (startIndex < content.length) {
            // For simplicity, assume that token counts are 75% of characters. (ChatGPT Reccomendation)
            let endIndex = Math.min(startIndex + maxTokensPerBatch * 3, content.length);
            let slice = content.slice(startIndex, endIndex);
            startIndex = endIndex;
            
            try{
                let sliceWithBounds = this.fileUtilService.addFileBoundaryMarkers(fileID, slice)
                let sliceTokenCount = this.encode.encode(sliceWithBounds).length;
    
                this.addFileToBatch(currentBatch, sliceWithBounds, sliceTokenCount);
                this.pushCurrentBatch(batches, currentBatch, prompt);
            }

            catch(e){
                console.log(e);
            }
        }
    }
    
    async getParsedResults(files) {
        for (const file of files) {
            await this.fileUtilService.parseJavaFile(file, this.parsedResults);
        }
    }

    // Used for testing, just sends one file at a time
    async simpleBatching(files, prompt, type) {
        let batchesOfText = []; // Array to hold all batches of text
        let filesPerBatch = []; // Used later on by the progress bar
        let id = 0;
        
        if (Object.keys(this.parsedResults).length === 0) {
            await this.getParsedResults(files);
        }
    
        for (const file of files) {
            // 
            let fullID = "ID-" + id.toString();
            this.idToNameMapping.set(fullID, path.basename(file));
            const fileContent = await this.fileUtilService.processJavaFile(file, fullID);

            switch (type) {
                case 'variables':
                    let variables = this.parsedResults[path.basename(file)]['variables'];
                    const variablesText = "\nHere are all the variables for this file:\n" + variables.join('\n');
                    batchesOfText.push(prompt + variablesText + this.fileUtilService.addFileBoundaryMarkers(fullID, fileContent));
                    // console.log(variablesText)    
                    break;
                case 'strings':
                    let strings = this.parsedResults[path.basename(file)]['strings'];
                    const stringsText = "\nHere are all the strings for this file:\n" + strings.join('\n');
                    batchesOfText.push(prompt + stringsText + this.fileUtilService.addFileBoundaryMarkers(fullID, fileContent));
                    // console.log(stringsText)  
                    break;
                case 'comments':
                    let comments = this.parsedResults[path.basename(file)]['comments'];
                    const commentsText = "\nHere are all the comments for this file:\n" + comments.join('\n');
                    batchesOfText.push(prompt + commentsText + this.fileUtilService.addFileBoundaryMarkers(fullID, fileContent));
                    // console.log(commentsText)
                    break;
                
                default:
                    batchesOfText.push(prompt + await this.fileUtilService.addFileBoundaryMarkers(fullID, fileContent));
            }
            filesPerBatch.push(1);
            id += 1;
        }

        return { batchesOfText, filesPerBatch };
    }

    async getCostEstimate(projectPath: string) {
        /*
        Results from previous runs to help estimate costs:
        CWEToyDataset: Predicted = $1.79, Actual = $1.23
        */
        const INPUT_COST = (5 / 1000000); // GPT-4o pricing is $5 per 1 million input tokens
        const OUTPUT_COST = (15 / 1000000); // GPT-4o pricing is $15 per 1 million output tokens
    
    
        // const prompts = [sensitiveVariablesPrompt, sensitiveStringsPrompt, sensitiveCommentsPrompt, sinkPrompt];
        const sourcePath = path.join(this.projectsPath, projectPath);
        const javaFiles = await this.fileUtilService.getJavaFilesInDirectory(sourcePath);
        let tokenCount = 0;
        
        const prompts = [
            { type: 'variables', prompt: sensitiveVariablesPrompt},
            { type: 'strings', prompt: sensitiveStringsPrompt},
            { type: 'comments', prompt: sensitiveCommentsPrompt},
            // { type: 'sinks', prompt: sinkPrompt}
        ];
    
        // Dictionary to store results by file name
        const fileResults = {};
    

        // Calculate the cost of each prompt concurrently
        const promptResults = await Promise.all(
            prompts.map(async (prompt) => {
                // let results = await this.dynamicBatching(javaFiles, prompt);
                let results = await this.simpleBatching(javaFiles, prompt.prompt, prompt.type);
                return results.batchesOfText;
            })
        );
    
        // Flatten the array of batches and calculate total tokens
        const batches = promptResults.flat();
        tokenCount = batches.reduce((total, batch) => total + this.encode.encode(batch).length, 0);
    
        let inputCost = tokenCount * INPUT_COST;
        let outputCost = tokenCount * OUTPUT_COST * 0.30; // Based on previous runs Output tokens are around 30% of input tokens
        let totalCost = inputCost + outputCost;
        this.idToNameMapping.clear();
        return {
            totalCost: totalCost,
            tokenCount: tokenCount,
            inputCost: inputCost,
            totalFiles: javaFiles.length,
        };
    }

    getChatGptToken() {
        const apiKey = this.configService.get('API_KEY');
        if (!apiKey) {
            return { message: 'API Key not set' };
        }
        const obfuscatedToken = apiKey.slice(0, -4).replace(/./g, '*') + apiKey.slice(-4);
        return { token: obfuscatedToken };
    }

    async updateChatGptToken(newToken: string) {
        const envPath = path.resolve(__dirname, '../../../.env'); // Path to your .env file
        let envContent = await this.fileUtilService.readFileAsync(envPath);
        
        // Update the API_KEY value in the .env file
        const updatedEnvContent = envContent.replace(
            /API_KEY='.*'(\r\n|\n)/,
            `API_KEY='${newToken}'$1`
        );
        // Write the updated content back to the .env file
        this.fileUtilService.writeToFile(envPath, updatedEnvContent);

        // Optionally reload the config values if needed
        this.configService['envVariablesLoaded'] = false;
        this.configService['load']();

        return { message: 'API Key updated successfully' };
    }

    replaceIDs(inputString) {
        return inputString.replace(/\b(ID-\d+)\.java\b/g, (match, id) => {
            // Logging the match to see what is being caught by the regex
            // console.log(`Found ID: ${match}`);
            if (this.idToNameMapping.get(id)) {
            //   console.log(`Replacing ${id} with ${this.idToNameMapping.get(id)}`);
              return this.idToNameMapping.get(id)
            } 
          });
        }
    

}

interface SensitiveVariables {
    name: string;
    description: string;
}

interface SinkType {
    name: string;
    type: string;
}

interface VariableObject {
    variable: string;
  }

interface StringObject {
    string: string;
}

interface CommentObject {
    comment: string;
}

interface JavaParseResult {
    filename: string;
    variables: string[];
    comments: string[];
    strings: string[];
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

