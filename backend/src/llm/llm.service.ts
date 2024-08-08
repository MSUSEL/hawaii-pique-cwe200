import { Injectable } from '@nestjs/common';
import OpenAI from "openai";
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import { Ollama } from 'ollama-node';
import axios from 'axios';
import { variablesPrompt } from './prompts/variablesPrompt';
import { stringsPrompt } from './prompts/stringsPrompt';
import { commentsPrompt } from './prompts/commentsPrompt';
import { sinksPrompt } from './prompts/sinksPrompt';

import { VariableParser, StringParser, CommentParser, SinkParser } from '../chat-gpt/JSON-parsers';
import async from 'async';

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

    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
    ) {
        this.projectPath = "";
        this.parsedResults = {};
        this.fileContents = {};
        this.contextMap = {};
    }

    async llmWrapper(filePaths: string[], sourcePath: string) {
        this.projectPath = sourcePath;
        await this.getParsedResults(filePaths);
        await this.readFiles(filePaths);

        const contextPromises = [];
        for (let fileName in this.fileContents) {
            contextPromises.push(this.getContext(fileName, this.parsedResults[fileName]));
        }

        await Promise.all(contextPromises);

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

        // Send all prompts to LLM concurrently and process responses
        const llamaPromises = [];
        for (let category of prompts) {
            for (let [fileName, promptMap] of category.input) {
                for (let [valueName, prompt] of promptMap) {
                    llamaPromises.push(this.callLlama(prompt).then(response => {
                        console.log(`Results for ${category.type}: ${response}`);

                        // Process the response
                        let jsonResponse = JSON.parse(response);
                        let file = jsonResponse.files[0];
                        if (file && file.fileName === fileName) {
                            let item = file[category.type][0]; // Assume only one entry per response
                            if (item.name === valueName) {
                                const isSensitive = category.type === 'sinks' ? item.isSink : item.isSensitive;
                                if (isSensitive === 'yes') {
                                    // Save data for JSON output (data.json)
                                    if (category.type === 'sinks') {
                                        JSONOutput[file.fileName][category.type].push({ name: item.name, type: item.type, reason: item.reason });
                                    } else {
                                        JSONOutput[file.fileName][category.type].push({ name: item.name, reason: item.reason });
                                    }
                                    // Save data as a mapping for YAML file, used for CodeQL
                                    category.parser.saveToMapping(category.mapping, file.fileName, file);
                                    // Save data as a list
                                    category.result.push(item.name);
                                }
                            }
                        }

                        // Emit progress to clients
                        this.eventsGateway.emitDataToClients('GPTProgress-' + category.type, JSON.stringify({
                            type: 'GPTProgress-' + category.type,
                            GPTProgress: Math.floor((category.result.length / category.input.size) * 100)
                        }));
                    }).catch(error => {
                        console.error(`Error processing ${category.type} response:`, error);
                    }));
                }
            }
        }

        await Promise.all(llamaPromises);

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

    async getParsedResults(filePaths: string[]) {
        let completed: number = 0;
        let total: number = filePaths.length;
        for (let filePath of filePaths) {
            await this.fileUtilService.parseJavaFile(filePath, this.parsedResults);
            completed += 1;
            let progressPercent = Math.floor((completed / total) * 100);
            this.eventsGateway.emitDataToClients('parsingProgress', JSON.stringify({ type: 'parsingProgress', parsingProgress: progressPercent }));
        }
    }

    async readFiles(filePaths: string[]) {
        let readPromises = filePaths.map(async (filePath) => {
            this.fileContents[path.basename(filePath)] = await this.fileUtilService.readFileAsync(filePath);
        });
        await Promise.all(readPromises);
    }

    async getContext(fileName: string, parsedValues: JavaParseResult) {
        const types = ['variables', 'comments', 'strings', 'sinks'];

        for (let type of types) {
            let items = parsedValues[type];
            let fileContents = await this.fileContents[fileName];

            for (let item of items) {
                let context = "";
                // Get all of the lines where the item is mentioned to be used as the context
                for (let line of fileContents.split('\n')) {
                    if (line.includes(item)) {
                        context += line + " ";
                    }
                }
                this.addToContextMap(fileName, type, item, context);
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
                            let prompt = ''
                            let singularType = type.substring(0, type.length - 1);

                            switch (type) {
                                case 'variables':
                                    promptTemplate = variablesPrompt;
                                    prompt = `Determine if the ${singularType} "${valueName}" in file "${fileName}" is sensitive:\n\n${context}\n\n${promptTemplate} \n\n Once again this is only for the ${singularType} "${valueName}" in file "${fileName}"`;
                                    break;
                                case 'strings':
                                    promptTemplate = stringsPrompt;
                                    prompt = `Determine if the ${singularType} "${valueName}" in file "${fileName}" is sensitive:\n\n${context}\n\n${promptTemplate} \n\n Once again this is only for the ${singularType} "${valueName}" in file "${fileName}"`;
                                    break;
                                case 'comments':
                                    promptTemplate = commentsPrompt;
                                    prompt = `Determine if the ${singularType} "${valueName}" in file "${fileName}" is sensitive:\n\n${promptTemplate} \n\n Once again this is only for the ${singularType} "${valueName}" in file "${fileName}"`;
                                    break;
                                case 'sinks':
                                    promptTemplate = sinksPrompt;
                                    prompt = `Determine if the ${singularType} "${valueName}" in file "${fileName}" is a sink:\n\n${context}\n\n${promptTemplate} \n\n Once again this is only for the ${singularType} "${valueName}" in file "${fileName}"`;
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
        // const llamaEndPoint = 'http://localhost:11434/api/generate';
        const llamaEndPoint = 'http://128.171.215.14:5000/generate';

        const requestBody = {
            "model": "llama3.1:latest",
            "prompt": prompt,
            "format": "json",
            "stream": false,
        };

        try {
            const response = await axios.post(llamaEndPoint, requestBody);
            console.log(response.data.response);
            return response.data.response;
        } catch (err) {
            console.error(err);
            throw err; // Re-throw the error to handle it in the calling function
        }
    }

    extractAndParseJSON(responseMessage: string): any {
        // Use regex or a JSON parser to extract and parse JSON from the response message
        // Example:
        const jsonMatch = responseMessage.match(/\{.*\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        }
        return {};
    }
}

interface JavaParseResult {
    filename: string;
    variables: string[];
    comments: string[];
    strings: string[];
    sinks: string[];
}
