import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import { exec, spawn } from 'child_process';
import { CodeQlParserService } from './codql-parser-service';
import * as path from 'path';
import * as fs from 'fs';
import { ConfigService } from '@nestjs/config';
import { ChatGptService } from 'src/chat-gpt/chat-gpt.service';
import { SensitiveVariablesContents } from './data';
import { SensitiveVariables } from './SensitiveVariables';
import { SensitiveComments } from './SensitiveComments';
import { SensitiveStrings } from './SensitiveStrings';
import { Sinks } from './Sinks';

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
        private gptService: ChatGptService
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
        // let slice = javaFiles.slice(177, 187);  
        // const data = await this.runChatGPT(slice, sourcePath);

        const data = await this.runChatGPT(javaFiles, sourcePath);

        await this.saveSensitiveInfo(data); // Saves all the sensitive info to .yml files

        await this.codeqlProcess(sourcePath, createCodeQlDto); // Creates a codeql database and runs the queries

        return await this.parserService.getSarifResults(sourcePath);

    }

    async runChatGPT(javaFiles, sourcePath){
        // Get Sensitive variables from gpt
        const data = await this.gptService.openAiGetSensitiveVariables(javaFiles);

        // Write response to file
        await this.writeFilesGptResponseToJson(data.fileList, sourcePath);

        return data;
    }

    useSavedData(sourcePath){
        return this.fileUtilService.parseJSONFile(path.join(sourcePath, "data.json"));
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


    formatMappings(mapping) : string{
        // mapping = {"GOOD_ConsistentAuthenticationTiming.java": ["VALID_USERNAME", "VALID_PASSWORD"], "GOOD_UniformLoginResponse": ["username"]};
        
        let result = "";

        // Iterate over each key (filename) in the mapping object
        Object.keys(mapping).forEach(key => {
            // For each variable associated with the key, generate a new line in the output
            mapping[key].forEach(variable => {
                result += `    - ["${key}", ${variable}]\n`;
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
            result += `    - ["${key}", ${sink[0]}, ${sink[1]}]\n`;
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
      
        const variablesMapping = this.formatMappings(data.sensitiveVariablesMapping);
        const stringsMapping = this.formatMappings(data.sensitiveStringsMapping);
        const commentsMapping = this.formatMappings(data.sensitiveCommentsMapping);
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

    async codeqlProcess(sourcePath: string, createCodeQlDto: any){
        // Remove previous database if it exists
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
        await this.fileUtilService.removeDir(db);

        // Create new database with codeql
        const createDbCommand = `database create ${db} --language=java --source-root=${sourcePath}`;
        console.log(createDbCommand);
        await this.runChildProcess(createDbCommand);

        // Analyze with codeql
        const extension = createCodeQlDto.extension ? createCodeQlDto.extension : 'sarif';
        const format = createCodeQlDto.format ? createCodeQlDto.format : 'sarifv2.1.0';

        const outputPath = path.join(sourcePath, `result.${extension}`);
        const analyzeDbCommand = `database analyze ${db} --format=${format} --rerun --output=${outputPath} ${this.queryPath}`;
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
}


