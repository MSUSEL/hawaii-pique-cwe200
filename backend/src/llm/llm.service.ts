import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import axios from 'axios';
import { variablesPrompt } from './prompts/variablesPrompt';
import { stringsPrompt } from './prompts/stringsPrompt';
import { commentsPrompt } from './prompts/commentsPrompt';
import { sinksPrompt } from './prompts/sinksPrompt';
import { VariableParser, StringParser, CommentParser, SinkParser } from '../chat-gpt/JSON-parsers';
import { Semaphore } from 'async-mutex'; // Import the Semaphore class
import { shuffle } from 'lodash'; // Import lodash for shuffling
import { JavaParserService } from 'src/parsers/implementations/java-parser.service';

@Injectable()
export class LLMService {
    projectPath: string;
    encode: any;
    parsedResults: { [key: string]: JavaParseResult };
    fileContents = {};
    contextMap = {};
    variablesInput = new Map<string, Map<string, string>>();
    stringsInput = new Map<string, Map<string, string>>();
    commentsInput = new Map<string, Map<string, string>>();
    sinksInput = new Map<string, Map<string, string>>();
    semaphore: Semaphore; // Declare the semaphore

    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
        private javaParserService: JavaParserService,
    ) {
        this.semaphore = new Semaphore(2); // Initialize the semaphore with a max concurrency of 2
    }

    async llmWrapper(filePaths: string[], sourcePath: string) {
        this.projectPath = sourcePath;
        // Parser the java files for variables, strings, comments, and method calls
        this.parsedResults = await this.javaParserService.getParsedResults(filePaths);

        // Read the files to get the context
        // await this.readFiles(filePaths);
        const contextPromises = [];
        for (let fileName in this.fileContents) {
            contextPromises.push(this.getContext(fileName, this.parsedResults[fileName]));
        }

        await Promise.all(contextPromises);

        // Create the prompts for the LLM that use the parsed results and the context
        let variables: string[] = [], strings: string[] = [], comments: string[] = [], sinks: string[] = [];
        const fileList: any[] = [];
        const JSONOutput = {};

        let sensitiveVariablesMapping = new Map<string, string[]>();
        let sensitiveStringsMapping = new Map<string, string[]>();
        let sensitiveCommentsMapping = new Map<string, string[]>();
        let sinksMapping = new Map<string, string[][]>();

        const prompts = [
            { type: 'variables', mapping: sensitiveVariablesMapping, result: variables, input: this.variablesInput, parser: new VariableParser() },
            { type: 'strings', mapping: sensitiveStringsMapping, result: strings, input: this.stringsInput, parser: new StringParser() },
            { type: 'comments', mapping: sensitiveCommentsMapping, result: comments, input: this.commentsInput, parser: new CommentParser() },
            { type: 'sinks', mapping: sinksMapping, result: sinks, input: this.sinksInput, parser: new SinkParser() }
        ];

        // Initialize JSONOutput with empty arrays for each file and type
        for (let fileName in this.fileContents) {
            JSONOutput[fileName] = { fileName: fileName, variables: [], strings: [], comments: [], sinks: [] };
        }

        // Create all prompts first
        await Promise.all(prompts.map(category => this.createPrompts(category.input, category.type)));
        // this.generate_training_data(prompts);

        // Process prompts in batches
        for (let category of prompts) {
            const requests: Promise<void>[] = [];

            for (let [fileName, promptMap] of category.input) {
                for (let [valueName, prompt] of promptMap) {
                    requests.push(this.processPrompt(fileName, valueName, prompt, category, JSONOutput));
                }
            }

            // Process the requests with concurrency control
            await this.processWithConcurrencyControl(requests);
        }

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

    async processPrompt(fileName: string, valueName: string, prompt: string, category: any, JSONOutput: any) {
        // Use the semaphore to limit the number of concurrent requests
        await this.semaphore.acquire();
        try {
            const response = await this.callLlama(prompt);
            console.log(`Results for ${category.type}: ${response}`);

            // Process the response
            let jsonResponse = JSON.parse(response);
            let file = jsonResponse.files[0];
            if (file && file.fileName === fileName) {
                let item = file[category.type][0]; // Assume only one entry per response
                if (item.name === valueName) {
                    const isSensitive = category.type === 'sinks' ? item.isSink : item.isSensitive;
                    
                    // if (isSensitive === 'yes') {
                        // Save data for JSON output (data.json)
                        // if (category.type === 'sinks') {
                        //     JSONOutput[file.fileName][category.type].push({ name: item.name, type: item.type, reason: item.reason });
                        // } else {
                        //     JSONOutput[file.fileName][category.type].push({ name: item.name, reason: item.reason });
                        // }

                    if (category.type === 'sinks') {
                        JSONOutput[file.fileName][category.type].push({ name: item.name, type: item.type, isSink: item.isSink, reason: item.reason });
                    } else {
                        JSONOutput[file.fileName][category.type].push({ name: item.name, isSensitive: item.isSensitive, reason: item.reason });
                    }

                    // Save data as a mapping for YAML file, used for CodeQL
                    category.parser.saveToMapping(category.mapping, file.fileName, file);
                    // Save data as a list
                    category.result.push(item.name);
                    // }
                }
            }

            // Emit progress to clients
            this.eventsGateway.emitDataToClients('GPTProgress-' + category.type, JSON.stringify({
                type: 'GPTProgress-' + category.type,
                GPTProgress: Math.floor((category.result.length / category.input.size) * 100)
            }));
        } catch (error) {
            console.error(`Error processing ${category.type} response:`, error);
        } finally {
            // Release the semaphore after processing
            this.semaphore.release();
        }
    }

    async processWithConcurrencyControl(requests: Promise<void>[]) {
        for (let i = 0; i < requests.length; i++) {
            await requests[i];
            
            // Throttle the requests to avoid rate limiting
            await new Promise((resolve) => setTimeout(resolve, 300));

        }
    }


    async readFiles(filePaths: string[]) {
        let readPromises = filePaths.map(async (filePath) => {
            this.fileContents[path.basename(filePath)] = await this.fileUtilService.readFileAsync(filePath);
        });
        await Promise.all(readPromises);
    }

    async getContext(fileName: string, parsedValues: JavaParseResult, contextType='Reduced') {
        const types = ['variables', 'comments', 'strings', 'sinks'];

        for (let type of types) {
            let items = parsedValues[type].map(item => item.name);

            
            // Read the whole files context

            switch (contextType) {
                // The full context is the whole file, which is what GPT used to use.
                case 'Full':
                    let fileContext = await this.fileContents[fileName];
                    // Includes all the items per for each type per prompt
                    this.addToContextMap(fileName, type, items, fileContext);
                    break;
            
                // The reduced context comes from the Java Parser service, similar to what BERT uses. 
                case 'Reduced':
                    for (let item of parsedValues[type]) {
                        let contextArrayMap = item['methods']
                        let itemContext = ''
                        for (let method of contextArrayMap) {
                            itemContext += parsedValues['methodCodeMap'][method]
                        }
                        console.log(item['name'], type, fileName);
                        // Includes one item per prompt
                        this.addToContextMap(fileName, type, item['name'], itemContext);
                    } 
                break;
                }
        }
    }

    addToContextMap(fileName: string, type: string, valueName: string, context: string) {
        if (!this.contextMap[fileName]) {
            this.contextMap[fileName] = {};
        }
        if (!this.contextMap[fileName][type]) {
            this.contextMap[fileName][type] = {};
        }
        this.contextMap[fileName][type][valueName] = context.trim();
    }

    async createPrompts(prompts: Map<string, Map<string, string>>, type: string) {
        for (const fileName in this.contextMap) {
            if (this.contextMap.hasOwnProperty(fileName)) {
                const fileTypes = this.contextMap[fileName];

                if (fileTypes.hasOwnProperty(type)) {
                    const values = fileTypes[type];

                    for (const valueName in values) {
                        if (values.hasOwnProperty(valueName)) {
                            const context = values[valueName];
                            let promptTemplate = '';
                            let prompt = '';
                            let singularType = type.substring(0, type.length - 1);

                            switch (type) {
                                case 'variables':
                                    promptTemplate = variablesPrompt;
                                    prompt = `Please carefully review the ${singularType} "${valueName}" in file "${fileName}" to determine if it is sensitive. Here is the context in which it is used:\n\n${context}\n\n${promptTemplate}\n\nIf the ${singularType} falls into any of the categories provided, respond with 'yes'. If unsure, please provide your reasoning and lean towards caution.`;
                                    break;
                                case 'strings':
                                    promptTemplate = stringsPrompt;
                                    prompt = `Determine if the ${singularType} "${valueName}" in file "${fileName}" is sensitive:\n\n${context}\n\n${promptTemplate} \n\n Once again this is only for the ${singularType} "${valueName}" in file "${fileName}". The main rule is if the string has actual hard-coded sensitive information, it is sensitive. If it is a generic string like "hello world" or "foo bar", it is not sensitive. If you are unsure, please respond with 'no'.`;
                                    break;
                                case 'comments':
                                    promptTemplate = commentsPrompt;
                                    prompt = `Determine if the ${singularType} "${valueName}" in file "${fileName}" is sensitive:\n\n${promptTemplate} \n\n Once again this is only for the ${singularType} "${valueName}" in file "${fileName}".`;
                                    break;
                                case 'sinks':
                                    promptTemplate = sinksPrompt;
                                    prompt = `Determine if the method "${valueName}" in file "${fileName}" is a sink:\n\n${context}\n\n${promptTemplate} \n\n Once again this is only for the method "${valueName}" in file "${fileName}".`;
                                    break;
                            }
                            // Store the prompt in the nested map
                            if (!prompts.has(fileName)) {
                                prompts.set(fileName, new Map<string, string>());
                            }
                            prompts.get(fileName).set(valueName, prompt);
                        }
                    }
                }
            }
        }
    }

    async callLlama(prompt: string) {
        const llamaEndPoint = 'http://128.171.215.14:5000/generate';

        const requestBody = {
            "model": "llama3.1:latest",
            "prompt": prompt,
            "format": "json",
            "stream": false,
        };

        try {
            const response = await axios.post(llamaEndPoint, requestBody);
            return response.data.response;
        } catch (err) {
            console.error(err);
            throw err; // Re-throw the error to handle it in the calling function
        }
    }

    extractAndParseJSON(responseMessage: string): any {
        // Use regex or a JSON parser to extract and parse JSON from the response message
        const jsonMatch = responseMessage.match(/\{.*\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        }
        return {};
    }

    async generate_training_data(prompts) {
        const labeledData = await this.loadLabeledData();
        const jsonlData = [];
    
        // Counters for positive and negative samples
        const sampleCounters = {
            variables: { positive: 0, negative: 0 },
            strings: { positive: 0, negative: 0 },
            comments: { positive: 0, negative: 0 },
            sinks: { positive: 0, negative: 0 },
        };
    
        for (let category of prompts) {
            const { type, input } = category;
    
            for (let [fileName, promptMap] of input) {
                for (let [valueName, prompt] of promptMap) {
                    const output = this.findMatchingLabel(labeledData, fileName, type, valueName);
    
                    // Track positive and negative samples
                    if (output) {
                        sampleCounters[type].positive++;
                    } else {
                        sampleCounters[type].negative++;
                    }
    
                    // Create a JSON object with the prompt and stringified output
                    const jsonlEntry = {
                        input: prompt,
                        output: JSON.stringify(this.formatOutput(fileName, type, valueName, output)).replace(/\\n/g, "\\n").replace(/\\r/g, "\\r")
                    };
    
                    jsonlData.push(jsonlEntry);
                }
            }
        }

        // Shuffle the data to ensure random distribution
        const shuffledData = shuffle(jsonlData);

        // Split the shuffled data into training (80%), validation (10%), and testing (10%) sets
        const totalData = shuffledData.length;
        const trainSize = Math.floor(totalData * 0.8);
        const validSize = Math.floor(totalData * 0.1);
        
        const trainData = shuffledData.slice(0, trainSize);
        const validData = shuffledData.slice(trainSize, trainSize + validSize);
        const testData = shuffledData.slice(trainSize + validSize);

        // Save the JSONL data to files
        // this.fileUtilService.writeToFile(path.join(this.projectPath, 'training_data.jsonl'), trainData.map(entry => JSON.stringify(entry)).join('\n'));
        // this.fileUtilService.writeToFile(path.join(this.projectPath, 'validation_data.jsonl'), validData.map(entry => JSON.stringify(entry)).join('\n'));
        // this.fileUtilService.writeToFile(path.join(this.projectPath, 'testing_data.jsonl'), testData.map(entry => JSON.stringify(entry)).join('\n'));
    
        // Print the counts of positive and negative samples
        console.log('Sample Summary:');
        for (const type in sampleCounters) {
            console.log(`${type}: ${sampleCounters[type].positive} positive, ${sampleCounters[type].negative} negative`);
        }
    }
    
    findMatchingLabel(labeledData, fileName, type, valueName) {
        const fileData = labeledData.find(data => data.fileName === fileName);
        if (!fileData) return null;
    
        const typeData = fileData[type];
        if (!typeData) return null;
    
        const matchedItem = typeData.find(item => item.name === valueName);
        return matchedItem || null;
    }
    
    formatOutput(fileName, type, valueName, output) {
        const isSensitiveOrSink = output ? "yes" : "no";
        const reason = output?.description || "No description provided.";
    
        // Start with the common structure
        let formattedOutput = {
            files: [
                {
                    fileName: fileName
                }
            ]
        };
    
        // Dynamically add the appropriate type
        const typeEntry = {};
        switch (type) {
            case 'variables':
                typeEntry['variables'] = [
                    {
                        name: valueName,
                        isSensitive: isSensitiveOrSink,
                        reason: reason
                    }
                ];
                break;
    
            case 'strings':
                typeEntry['strings'] = [
                    {
                        name: valueName,
                        isSensitive: isSensitiveOrSink,
                        reason: reason
                    }
                ];
                break;
    
            case 'comments':
                typeEntry['comments'] = [
                    {
                        name: valueName,
                        isSensitive: isSensitiveOrSink,
                        reason: reason
                    }
                ];
                break;
    
            case 'sinks':
                typeEntry['sinks'] = [
                    {
                        name: valueName,
                        isSink: isSensitiveOrSink,
                        type: output?.type || "Unknown type",
                        reason: reason
                    }
                ];
                break;
    
            default:
                throw new Error(`Unknown type: ${type}`);
        }
    
        // Merge the type-specific entry into the formatted output
        Object.assign(formattedOutput.files[0], typeEntry);
    
        return formattedOutput;
    }
    
    async loadLabeledData() {
        const labeledDataPath = path.join(this.projectPath, 'labeledData.json');
        const labeledDataContent = await this.fileUtilService.readFileAsync(labeledDataPath);
        return JSON.parse(labeledDataContent);
    }
}

interface JavaParseResult {
    filename: string;
    variables: string[];
    comments: string[];
    strings: string[];
    sinks: string[];
}
