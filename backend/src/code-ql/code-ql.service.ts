import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import { exec, spawn } from 'child_process';
import { CodeQlParserService } from './codql-parser-service';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

import { ConfigService } from '@nestjs/config';
import { ChatGptService } from 'src/chat-gpt/chat-gpt.service';
import { SensitiveVariablesContents } from './data';
import { SensitiveVariables } from './SensitiveVariables';
import { SensitiveComments } from './SensitiveComments';
import { SensitiveStrings } from './SensitiveStrings';
import { Sinks } from './Sinks';
import { BertService } from 'src/bert/bert.service';
import { LLMService } from 'src/llm/llm.service';
import { Region } from './codql-parser-service';
import { FlowNode } from './codql-parser-service';

import { EventsGateway } from 'src/events/events.gateway';
@Injectable()
export class CodeQlService {
    projectsPath: string;
    queryPath: string;
    constructor(
        private configService: ConfigService,
        private parserService: CodeQlParserService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
        private gptService: ChatGptService,
        private bertService: BertService,
        private llmService: LLMService
    ) {
        this.projectsPath = this.configService.get<string>(
            'CODEQL_PROJECTS_DIR',
        );

        this.queryPath = path.join(
            // '../',
            this.configService.get<string>('QUERY_DIR'),
            'codeql-custom-queries-java',
        );
    }

    /**
     * Run codeql query against a target project
     *
     * @param createCodeQlDto Data transfer object with project name
     */
    async runCodeQl(createCodeQlDto: any) {
        // Get all java files in project
        const sourcePath = path.join(this.projectsPath, createCodeQlDto.project);
        const javaFiles = await this.fileUtilService.getJavaFilesInDirectory(sourcePath);
    
        // await this.runBert(javaFiles, sourcePath, createCodeQlDto);

        if (createCodeQlDto.extension === 'csv'){
            return await this.parserService.getcsvResults(sourcePath);
        }
        return await this.parserService.getSarifResults(sourcePath);

    
    }

    async runChatGPT(sourcePath){
        // Get Sensitive variables from gpt
        const data = await this.gptService.LLMWrapper(sourcePath);

        // Write response to file
        await this.writeFilesGptResponseToJson(data.fileList, sourcePath);

        return data;
    }

    async runBert(javaFiles, sourcePath, createCodeQlDto: any) {
        const times = {};
    
        // Helper function to format time in minutes and seconds
        const formatTime = (seconds: number) => {
            if (seconds > 60) {
                const minutes = Math.floor(seconds / 60); // Get the whole minutes
                const remainingSeconds = Math.floor(seconds % 60); // Get the remaining seconds
                return `${minutes} minute${minutes > 1 ? 's' : ''} and ${remainingSeconds} second${remainingSeconds > 1 ? 's' : ''}`;
            } else {
                return `${Math.floor(seconds)} second${seconds > 1 ? 's' : ''}`; // Show seconds
            }
        };

        async function executeStep(stepName, stepFunction) {
            console.log('-----------------------------------')
            console.log(stepName)
            console.log('-----------------------------------')


            try {
                await recordTime(stepName, stepFunction);
            } catch (error) {
                console.error(`Error in ${stepName}:`, error);
                // Notify the client about the error
                console.log(`Error in ${stepName}: ${error.message}`);
                // Stop further execution by re-throwing the error
                throw error(error.message);
            }
        }
    
        // Helper function to record the time taken for each step
        const recordTime = async (stepName: string, fn: () => Promise<void>) => {
            const start = Date.now();
            await fn();
            const end = Date.now();
            times[stepName] = (end - start) / 1000; // Time in seconds
        };

        await executeStep('Parsing files for variables, strings, comments, and method calls.', async () => {
            await this.bertService.bertWrapper(javaFiles, sourcePath);
        });
    
        await executeStep('Detecting sensitive info using BERT.', async () => {
            await this.bertService.getBertResponse(sourcePath, "run_bert.py");
        });
        let data = null;
        await executeStep('Reading in BERT results from data.json.', async () => {
            data = this.useSavedData(sourcePath);
        });
    
        await executeStep('Saving the sensitive info to .yml files.', async () => {
            await this.saveSensitiveInfo(data);
        });
    
        await executeStep('Creating CodeQL database.', async () => {
            await this.createDatabase(sourcePath, createCodeQlDto);
        });
    
        // await executeStep('Running the backward slice queries.', async () => {
        //     await this.performBackwardSlicing(sourcePath, createCodeQlDto);
        // });
    
        // await executeStep('Parsing the backward slice graphs.', async () => {
        //     await this.bertService.parseBackwardSlice(sourcePath);
        // });
    
        // await executeStep('Running BERT with backward slice graphs.', async () => {
        //     await this.bertService.getBertResponse(sourcePath, 'bert_with_graph.py');
        // });
    
        // await executeStep('Updating sensitiveVariables.yml.', async () => {
        //     const sensitiveVariables = this.useSavedData(sourcePath, 'sensitiveVariables.json');
        //     this.saveUpdatedSensitiveVariables(sensitiveVariables);
        // });
    
        await executeStep('Running CWE queries.', async () => {
            await this.runCWEQueries(sourcePath, createCodeQlDto);
        });
    
        // Print all the times at the end
        console.log("Time taken for each step:");
        Object.keys(times).forEach(step => {
            console.log(`${step}: ${formatTime(times[step])}`);
        });
    }

    async runLLM(javaFiles, sourcePath){
        // Get Sensitive variables from gpt
        const data = await this.llmService.llmWrapper(javaFiles, sourcePath);
        await this.writeFilesGptResponseToJson(data.fileList, sourcePath);
        return data;

    }

    useSavedData(sourcePath, fileName = 'data.json'){
        return this.fileUtilService.parseJSONFile(path.join(sourcePath, fileName));
    }


    /**
     * Execute command string using codeql
     *
     * @param codeQlCommand formatted codeql arguments
     */
    runChildProcess(codeQlCommand: string): Promise<void> {
        const commands = codeQlCommand.split(' ');
        return new Promise((resolve, reject) => {
            // create new codeql process
            let childProcess = spawn('codeql', commands);

            // report stdout
            childProcess.stdout.on('data', (data) => {
                console.log(data.toString());
                this.eventsGateway.emitDataToClients('data', data.toString())
            });

            // report stderr
            childProcess.stderr.on('data', (data) => {
                console.log(data.toString());
                this.eventsGateway.emitDataToClients('data', data.toString())
            });

            // report results after finishing
            const self = this;
            childProcess.on('exit', function (code, signal) {
                const result = "process CodeQl exited with code " + code + " and signal " + signal;
                // if (code !== 0) {
                //     reject(result);
                //     throw new Error(result);
                // }
                console.log(result);
                self.eventsGateway.emitDataToClients('data', result.toString())
                resolve();
            });

            // report errors
            childProcess.on('error', (error) => {
                console.log(error);
                reject(error);
            });
        });
    }

    /**
     * Write variables to file
     *
     * @param variables variables to write
     */
    async writeVariablesToFile(variables: string, filePath: string) {
        // todo why is this a constant path? Can be moved to config or other?
        // const filePath = "../codeql/ql/java/ql/lib/semmle/code/java/security/SensitiveVariables.qll";
        // const filePath = "../codeql queries/SensitiveInfo.yml";

        await this.fileUtilService.writeToFile(filePath, variables)
    }

    /**
     * Write JSON string of GPT response to project root
     *
     * @param fileList list of files to include in the json report
     * @param sourcePath path of project root
     */
    async writeFilesGptResponseToJson(fileList: any[], sourcePath: string) {
        const jsonPath = path.join(sourcePath, "data.json");
        const data = JSON.stringify(fileList, null, '\t');    // additional args for pretty printing
        await this.fileUtilService.writeToFile(jsonPath, data)
    }


    formatMappings(mapping: { [key: string]: string[] }, type: string): string {
        let result = "";
    
        // Iterate over each key (filename) in the mapping object
        Object.keys(mapping).forEach(key => {
            // For each variable associated with the key, generate a new line in the output
            mapping[key].forEach(variable => {
                // Remove Unicode characters, double quotes, single quotes, and newlines from the variable
                // Ensure backslashes are escaped
                variable = variable.replace(/[^\x00-\x7F]/g, '').replace(/["'\n]/g, '').replace(/\\/g, '\\\\');
    
                // Apply exclusion rules
                if (
                    // Exclude common non-sensitive patterns
                    /example/i.test(variable) ||
                    /test/i.test(variable) ||
                    /demo/i.test(variable) ||
                    /foo/i.test(variable) ||
                    /bar/i.test(variable) ||
                    /baz/i.test(variable) ||
                    /secret/i.test(variable) ||
                    // Exclude empty strings
                    variable === "" ||
                    // Exclude whitespace-only strings
                    /^\s*$/.test(variable) ||
                    // Exclude strings with exactly one dot followed by a digit
                    /^[^.]*\.[0-9]+$/.test(variable)
                ) {
                    return; // Skip this variable if it matches any exclusion criteria
                }
    
                // Check if the variable is in the format {key: value, key: value}
                const regex = /{([^}]+)}/;
                const match = regex.exec(variable);
                if (match) {
                    // Extract values from the variable
                    const valuesString = match[1];
                    const values = valuesString.split(',').map(pair => {
                        const parts = pair.split(':');
                        return parts.length > 1 ? parts[1].trim() : '';
                    });
                    if (type === "variables") {
                        values.forEach(value => {
                            result += `    - ["${key}", ${value}]\n`;
                        });
                    } else {
                        values.forEach(value => {
                            result += `    - ["${key}", "${value}"]\n`;
                        });
                    }
                } else {
                    // If not in the format {key: value, key: value}, handle as normal string
                    if (type === "variables") {
                        result += `    - ["${key}", "${variable}"]\n`;
                    } else {
                        result += `    - ["${key}", "${variable}"]\n`;
                    }
                }
            });
        });
    
        return result;
    }    
    
    formatCommentsMapping(mapping: { [key: string]: string[] }, type): string {
        let result = "";
    
        // Iterate over each key (filename) in the mapping object
        Object.keys(mapping).forEach(key => {
            // For each variable associated with the key, generate a new line in the output
            mapping[key].forEach(variable => {
                result += `    - ["${key}", "${variable.replace(/(?!^)"(?!$)/g, '\\"')}"]\n`;
            });
        });
    
        return result;
    }
    
     
    formatSinkMappings(mapping: Map<string, string[][]>): string {
        let result = "";
        // Iterate over each key-value pair in the mapping object
        for (const [key, sinks] of mapping.entries()) {
          // For each array associated with the key, generate a new line in the output
          for (const sink of sinks) {
            result += `    - ["${key}", "${sink[0]}", "${sink[1]}"]\n`;
          }
        }
        return result;
      }
    formatStringArray(inputArray: string[]): string {
        // Initialize the result string
        let result = '';
    
        // Loop through each string in the array
        inputArray.forEach(item => {
            // Append the formatted item to the result string
            result += `    - [${item}]\n`;
        });
    
        return result;
    }

    async saveSensitiveInfo(data){
      
        const variablesMapping = this.formatMappings(data.sensitiveVariablesMapping, "variables");
        const stringsMapping = this.formatMappings(data.sensitiveStringsMapping, "strings");
        const commentsMapping = this.formatCommentsMapping(data.sensitiveCommentsMapping, "comments");
        const sinksMapping = this.formatSinkMappings(data.sinksMapping);
            
        let variablesFile = SensitiveVariables.replace("----------", variablesMapping);
        await this.writeVariablesToFile(variablesFile, "../codeql queries/SensitiveInfo/SensitiveVariables.yml")
        await this.writeVariablesToFile(variablesFile, "../codeql/codeql-custom-queries-java/SensitiveInfo/SensitiveVariables.yml")
        
 
        let stringsFile = SensitiveStrings.replace("++++++++++", stringsMapping);
        await this.writeVariablesToFile(stringsFile, "../codeql queries/SensitiveInfo/SensitiveStrings.yml")
        await this.writeVariablesToFile(stringsFile, "../codeql/codeql-custom-queries-java/SensitiveInfo/SensitiveStrings.yml")
        
  
        let commentsFile = SensitiveComments.replace("**********", commentsMapping);
        await this.writeVariablesToFile(commentsFile, "../codeql queries/SensitiveInfo/SensitiveComments.yml")
        await this.writeVariablesToFile(commentsFile, "../codeql/codeql-custom-queries-java/SensitiveInfo/SensitiveComments.yml")
        

        let sinksFile = Sinks.replace("----------", sinksMapping);
        await this.writeVariablesToFile(sinksFile, "../codeql queries/SensitiveInfo/Sinks.yml")
        await this.writeVariablesToFile(sinksFile, "../codeql/codeql-custom-queries-java/SensitiveInfo/Sinks.yml")
        

    }

    async saveUpdatedSensitiveVariables(data){
      
        const variablesMapping = this.formatMappings(data.sensitiveVariablesMapping, "variables");
        let variablesFile = SensitiveVariables.replace("----------", variablesMapping);
        await this.writeVariablesToFile(variablesFile, "../codeql queries/SensitiveInfo/SensitiveVariables.yml")
        await this.writeVariablesToFile(variablesFile, "../codeql/codeql-custom-queries-java/SensitiveInfo/SensitiveVariables.yml")

    }

    async createDatabase(sourcePath: string, createCodeQlDto: any){
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
        await this.fileUtilService.removeDir(db);
        const createDbCommand = `database create ${db} --language=java --source-root=${sourcePath}`;
        console.log(createDbCommand);
        await this.runChildProcess(createDbCommand);
    }

    async performBackwardSlicing(sourcePath: string, createCodeQlDto: any){
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
        const extension = 'sarif';
        const format = 'sarifv2.1.0';
        const outputFileName = 'backwardslice';
        const outputPath = path.join(sourcePath, `${outputFileName}.${extension}`);
        const threads = 12;
        const totalMemoryMB = os.totalmem() / (1024 * 1024);  // Total system memory in MB
        const ramAllocationMB = Math.floor(totalMemoryMB * 0.8);  // 80% of total memory
        const queryPath = path.join(this.queryPath, 'ProgramSlicing', 'Variables');

        // Command to run backward slicing
        const analyzeDbCommand = 
        `database analyze ${db} ` + 
        `--format=${format} --rerun ` + 
        `--output=${outputPath} ${queryPath} ` + 
        `--max-paths=1 --sarif-add-snippets=true ` +
        `--threads=${threads} --ram=${ramAllocationMB}`;
        await this.runChildProcess(analyzeDbCommand);

    }

    async runCWEQueries(sourcePath: string, createCodeQlDto: any) {
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
        const extension = 'sarif';
        const format = 'sarifv2.1.0';
        const outputFileName = 'result';
        const outputPath = path.join(sourcePath, `${outputFileName}.${extension}`);
        const threads = 12;
        const totalMemoryMB = os.totalmem() / (1024 * 1024);  // Total system memory in MB
        const ramAllocationMB = Math.floor(totalMemoryMB * 0.8);  // 80% of total memory
        const queryPath = this.queryPath;

        // This is for running all of the queries
            const queryDir = path.resolve(queryPath);
            const excludeDir = path.resolve(path.join(queryPath, 'ProgramSlicing'));
            
            function collectQlFiles(dir) {
                let results = [];
                const list = fs.readdirSync(dir);
                list.forEach(file => {
                    const filePath = path.join(dir, file);
                    const stat = fs.statSync(filePath);
                    if (stat && stat.isDirectory()) {
                        // Recursively collect .ql files from subdirectories
                        results = results.concat(collectQlFiles(filePath));
                    } else if (filePath.endsWith('.ql') && !filePath.startsWith(excludeDir)) {
                        // Include only .ql files and exclude those in the ProgramSlicing directory
                        results.push(filePath);
                    }
                });
                return results;
            }
            
            // Collect all .ql files from the queryDir, excluding ProgramSlicing subdir
            let queriesToRun = collectQlFiles(queryDir);
            
            // Join the selected queries into a single string
            const queryList = queriesToRun.join(' ');
            
            // Build the command with the filtered list of queries
            const analyzeDbCommand = 
            `database analyze ${db} ` +
            `--format=${format} ` + 
            `--rerun ` + 
            `--output=${outputPath} ${queryList} ` + 
            `--threads=${threads} ` + 
            `--ram=${ramAllocationMB}`;
            await this.runChildProcess(analyzeDbCommand);
    }

    async getSarifResults(project: string){
        const sourcePath = path.join(this.projectsPath, project);
        console.log(sourcePath);
        if (!fs.existsSync(path.join(sourcePath, 'result.sarif'))) {
            return { error: 'Results file does not exist' };
        }
        let data = await this.parserService.getSarifResults(sourcePath);
        return await this.parserService.getSarifResults(sourcePath);
    }

    async getDataFlowTree(vulnerabilityId: string, project: string, index: string){ {
        
        // Find the first occurrence of the project in the vulnerabilityId
        let projectIndex = vulnerabilityId.indexOf(project);

        // If the project is found, slice after the first occurrence of the project
        if (projectIndex !== -1) {
        // Slice from the first occurrence of the project
        vulnerabilityId = vulnerabilityId.slice(projectIndex);
        
        // If there is a second occurrence of the project, remove everything before it
        let secondProjectIndex = vulnerabilityId.indexOf(project, project.length);
        if (secondProjectIndex !== -1) {
            vulnerabilityId = vulnerabilityId.slice(0, secondProjectIndex - 1) + vulnerabilityId.slice(secondProjectIndex + project.length);
            }
        }
        
        // console.log(vulnerabilityId, project, index);

        const sourcePath = path.join(this.projectsPath, project);
        return await this.parserService.getDataFlowTree(vulnerabilityId, sourcePath, index);
    }
}
}

