import { Injectable } from '@nestjs/common';
import OpenAI from "openai";
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import {sensitiveVariablesPrompt} from './sensitiveVariablesPrompt';
import {sensitiveStringsPrompt} from './sensitiveStringsPrompt';
import {sensitiveCommentsPrompt} from './sensitiveCommentsPrompt';
import { sinkPrompt } from './sinkPrompt';
import { response } from 'express';
import { get_encoding } from 'tiktoken';
import async from 'async';
import { spawn } from 'child_process';
import { Ollama } from 'ollama-node';
import {VariableParser, StringParser, CommentParser, SinkParser} from './JSON-parsers'


@Injectable()
export class ChatGptService {
    openai: OpenAI;
    debug: string;
    projectsPath: string;
    encode: any;
    idToNameMapping: Map<string, string> = new Map<string, string>();
    parsedResults:{ [key: string]: JavaParseResult };
    
    variablesInput = new Map<string, string>();
    stringsInput = new Map<string, string>();
    commentsInput = new Map<string, string>();
    sinksInput = new Map<string, string>();

    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
    ) {
        const api_key = this.configService.get('API_KEY');
        this.openai = new OpenAI({apiKey: api_key,});

        this.projectsPath = this.configService.get<string>('CODEQL_PROJECTS_DIR',);
        this.debug = this.configService.get('DEBUG');
        this.encode = get_encoding("o200k_base");
        this.parsedResults = {};
    }

    /**
     * Process files into a standardized format to be appended to a GPT query
     *
     * @param files files to include in GPT prompt
     */
    async LLMWrapper(files: string[]) {
        let variables: string[] = [], strings: string[] = [], comments: string[] = [], sinks: string[] = [];
        const fileList: any[] = [];
        
        let sensitiveVariablesMapping = new Map<string, string[]>();
        let sensitiveStringsMapping = new Map<string, string[]>();
        let sensitiveCommentsMapping = new Map<string, string[]>();
        let sinksMapping = new Map<string, string[][]>();
        let rawResponses = "";
        
        const prompts = [
            { type: 'variables', mapping: sensitiveVariablesMapping, result: variables, input: this.variablesInput , parser: new VariableParser()},
            // { type: 'strings', mapping: sensitiveStringsMapping, result: strings, input: this.stringsInput, parser: new StringParser()},
            // { type: 'comments', mapping: sensitiveCommentsMapping, result: comments, input: this.commentsInput, parser: new CommentParser()},
            // { type: 'sinks', mapping: sinksMapping, result: sinks, input: this.sinksInput, parser: new SinkParser() }
        ];
        
        // Dictionary to store results by file name
        const JSONOutput = {};
    
        for (const { type, mapping, result, input, parser } of prompts) {
            let completedBatches = 0;
            const batches = Array.from(input.values());
            const filesPerBatch = Array.from(input.keys());
            const totalBatches = batches.length;
    
            const processBatch = async (batch: string, batch_number: number) => {
                try {
                    const response = await this.createGptWithBackoff(batch, batch_number);
                    rawResponses += this.replaceIDs(response.message);
                    console.log(`Results for batch ${batch_number} \n ${response.message}`);
                    completedBatches += 1;
    
                    this.eventsGateway.emitDataToClients('GPTProgress-' + type, JSON.stringify({ 
                        type: 'GPTProgress-' + type, 
                        GPTProgress: Math.floor((completedBatches / totalBatches) * 100) }
                    ));
  
                    let json = extractAndParseJSON(response.message);
    
                    json.files.forEach((file: any) => {
                        // let fileID = file.fileName.split('.java')[0];
                        // let fileName = this.idToNameMapping.get(fileID);

                        let fileName = file.fileName
                        
                        // Save data for JSON output (data.json)
                        parser.saveToJSON(JSONOutput, fileName, type, file);
                        // Save data as a mapping for YMAL file, used for CodeQL 
                        parser.saveToMapping(mapping, fileName, file);
                        // Save data as a list
                        result.push(...parser.getNamesAsList(file));

                    });
                } catch (error) {
                    console.error('Error processing GPT response:', error);
                }
            };
    
            const processConcurrentBatches = async (batches, filesPerBatch) => {
                let concurrencyLimit = 50; // Number of concurrent batches to run (Used so we don't overload the GPT API)
                console.log(`Finding ${type} in Project`);
                        
                const queue = async.queue(async (task, callback) => {
                    await processBatch(task.batch, task.index);
                    callback();
                }, concurrencyLimit);
            
                batches.forEach((batch, batch_number) => {
                    queue.push({ batch, files: filesPerBatch[batch_number], batch_number });
                });
            
                await queue.drain();
            };
    
            await processConcurrentBatches(batches, filesPerBatch);
        
            this.eventsGateway.emitDataToClients('GPTProgress-' + type, JSON.stringify({ 
                type: 'GPTProgress-' + type, GPTProgress: 100 }));
        }
    
        // Write raw responses to a file
        this.fileUtilService.writeToFile(path.join(this.projectsPath, 'rawResponses.txt'), rawResponses);
    
        // Wrap the dictionaries into arrays for the JSON output
        for (const fileName in JSONOutput) {
            fileList.push(JSONOutput[fileName]);
        }
        
        // Remove duplicates
        variables = [...new Set(variables)];
        strings = [...new Set(strings)];
        comments = [...new Set(comments)];
        sinks = [...new Set(sinks)];
    
        return { variables, strings, comments, sinks, fileList, sensitiveVariablesMapping, sensitiveStringsMapping, sensitiveCommentsMapping, sinksMapping };
    }
    
    
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
                        console.log(`Rate limit hit. Retrying batch ${index} in ${timeOut} seconds`)
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
        // console.log(prompt);
        try {
            // Break the prompt into sections, for better api usage
            // let sections = this.extractSections(prompt);

            const response = await this.openai.chat.completions.create({
                model: 'gpt-4o',
                // temperature: 0.0,
                // top_p: 0.05,
                messages: [
                    { role: 'system', content: prompt }, 
                    // { role: 'user', content: sections.values }, 
                    // { role: 'user', content: sections.code }
                ],
                response_format: { type: "json_object" },
            });

            return { message: response.choices[0].message.content };

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

    async getFileGptResponse(filePath: String) {
        var directories = filePath.split('\\');
        var jsonFilePath = path.join(directories[0], directories[1], 'data.json');
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
    
    async getParsedResults(files: string[]) {
        let completed: number = 0;
        let total: number = files.length;
        for (const file of files) {
            try{
                await this.fileUtilService.parseJavaFile(file, this.parsedResults);
            }
            catch(e){
                console.log(e); 
        }
        completed += 1;
        let progressPercent = Math.floor((completed / total) * 100);
        this.eventsGateway.emitDataToClients('parsingProgress', JSON.stringify({ type: 'parsingProgress', parsingProgress: progressPercent }));
    }
}

    // Used for testing, just sends one file at a time
    async simpleBatching(files: string[], prompt: string, type: string, output: Map<string, string>, progress: { value: number, total: number }) {
        let id: number = 0;
        for (const file of files) {
            // let fullID = "ID-" + id.toString();
            let fullID: string = path.basename(file);
            // this.idToNameMapping.set(fullID, path.basename(file));
            const fileContent = await this.fileUtilService.processJavaFile(file, fullID);

            console.log(fileContent);

            if (!this.parsedResults[path.basename(file)]){
                // console.log(`Parsing file ${file}`);
                await this.fileUtilService.parseJavaFile(file, this.parsedResults);
            }

            try {
                let variables:string[] = [], strings:string[] = [], comments:string[] = [];
                const baseFileName: string = path.basename(file);

                switch (type) {
                    case 'variables':
                        variables = this.parsedResults[baseFileName]['variables'] || [];
                        // const variablesText = "\n+++++\nI have already done all of the parsing for you, here are all the variables in this file - " + baseFileName +":\n" + variables.map((variable, index) => `${index + 1}. ${variable}`).join('\n') + "\n+++++\n";
                        const variablesText = "\n+++++\nI have already done all of the parsing for you, here are all the variables in this file - " + baseFileName + ":\n" + variables.join(', ') + "\n+++++\n";
                        output.set(baseFileName, prompt + variablesText + this.fileUtilService.addFileBoundaryMarkers(fullID, fileContent));
                        // console.log(output.get(baseFileName));
                        break;
                    case 'strings':
                        strings = this.parsedResults[baseFileName]['strings'] || [];
                        const stringsText = "\n+++++\nI have already done all of the parsing for you, here are all the strings in this file:\n" + strings.map((string, index) => `${index + 1}. ${string}`).join('\n') + "\n+++++\n";
                        output.set(baseFileName, prompt + stringsText + this.fileUtilService.addFileBoundaryMarkers(fullID, fileContent));
                        break;
                    case 'comments':
                        comments = this.parsedResults[baseFileName]['comments'] || [];
                        const commentsText = "\n+++++\nI have already done all of the parsing for you, here are all the comments in this file - " + baseFileName + ":\n" + comments.map((comment, index) => `${index + 1}. ${comment}`).join('\n') + "\n+++++\n";
                        // output.set(baseFileName, prompt + commentsText + this.fileUtilService.addFileBoundaryMarkers(fullID, fileContent));
                        output.set(baseFileName, prompt + commentsText);
                        // console.log(output.get(baseFileName));

                        break;
                    default:
                        output.set(baseFileName, prompt + await this.fileUtilService.addFileBoundaryMarkers(fullID, fileContent));
                }
            } catch (e) {
                console.error(`Failed to parse JSON for file ${file}: ${e.message}`);
            }

            // Update progress in a synchronized manner
            progress.value += 1;
            let progressPercent = Math.floor((progress.value / progress.total) * 100);
            this.eventsGateway.emitDataToClients('estimateProgress', JSON.stringify({ type: 'estimateProgress', estimateProgress: progressPercent }));

            id += 1;
        }
    }

    async getCostEstimate(projectPath: string) {
        const INPUT_COST = (5 / 1000000); // GPT-4o pricing is $5 per 1 million input tokens
        const OUTPUT_COST = (15 / 1000000) * .15; // GPT-4o pricing is $15 per 1 million output tokens

        // Get the source path and Java files
        const sourcePath = path.join(this.projectsPath, projectPath);
        const javaFiles: string[] = await this.fileUtilService.getJavaFilesInDirectory(sourcePath);

        // Parse the Java files to get the variables, strings, and comments
        await this.getParsedResults(javaFiles);

        let totalTokenCount = 0;

        const prompts = [
            { type: 'variables', prompt: sensitiveVariablesPrompt, output: this.variablesInput },
            { type: 'strings', prompt: sensitiveStringsPrompt, output: this.stringsInput },
            { type: 'comments', prompt: sensitiveCommentsPrompt, output: this.commentsInput },
            { type: 'sinks', prompt: sinkPrompt, output: this.sinksInput }
        ];

        // Clear previous outputs
        this.variablesInput.clear();
        this.stringsInput.clear();
        this.commentsInput.clear();
        this.sinksInput.clear();

        // Initialize shared progress state
        let progress = { value: 0, total: prompts.length * javaFiles.length };

        // Helper function to process each prompt
        const processPrompt = async (promptObj: { type: string; prompt: string; output: Map<string, string> }) => {
            let tokenCount = 0;
            await this.simpleBatching(javaFiles, promptObj.prompt, promptObj.type, promptObj.output, progress);

            // Calculate tokens for the current prompt
            const outputArray = Array.from(promptObj.output.values());
            for (const batch of outputArray) {
                totalTokenCount += this.encode.encode(batch).length;
                tokenCount += this.encode.encode(batch).length;
            }
            console.log(`Token count for ${promptObj.type}: ${(tokenCount).toFixed(2)}`);
        };

        // Calculate the cost of each prompt concurrently
        await Promise.all(prompts.map(prompt => processPrompt(prompt)));

        const result = javaFiles.reduce((acc, fileName) => {
            const baseName = path.basename(fileName);
            
            const fileResult: { [key: string]: any } = {};
            
            prompts.forEach(promptObj => {
                const output = promptObj.output.get(baseName) || '';
                fileResult[promptObj.type] = {
                    input: output,
                    output: ''
                };
            });
        
            acc[baseName] = fileResult;
            return acc;
        }, {} as { [key: string]: any });
        

        const outputFilePath = path.join(this.projectsPath, projectPath, 'output_prompts.json');
        this.fileUtilService.writeToFile(outputFilePath, JSON.stringify(result, null, 2));

        // Calculate cost
        const inputCost = totalTokenCount * INPUT_COST;
        const outputCost = totalTokenCount * OUTPUT_COST;
        const totalCost = inputCost + outputCost;

        console.log(`Estimated cost for project ${projectPath}: $${totalCost.toFixed(2)}`);
        return {
            totalCost: totalCost,
            tokenCount: totalTokenCount,
            inputCost: inputCost,
            totalFiles: javaFiles.length,
        };
    }
    extractSections(text: string): { prompt: string, values: string, code: string } {
        const regex = /^(.*?)\+{5}(.*?)\+{5}(.*)$/s;
        const match = regex.exec(text);
    
        if (match) {
            return {
                prompt: match[1].trim(),
                values: match[2].trim(),
                code: match[3].trim()
            };
        } else {
            throw new Error("The text format is not correct.");
        }
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
    // description: string;
}

interface SinkType {
    name: string;
    type: string;
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
