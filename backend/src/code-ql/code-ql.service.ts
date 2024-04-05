import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import { exec, spawn } from 'child_process';
import { CodeQlParserService } from './codql-parser-service';
import * as path from 'path';
import * as fs from 'fs';
import { ConfigService } from '@nestjs/config';
import { ChatGptService } from 'src/chat-gpt/chat-gpt.service';
import { SensitiveVariablesContents } from './data';
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

        // const data = this.debugChatGPT(sourcePath);
        // await this.runChatGPT(sourcePath);



        const javaFiles = await this.fileUtilService.getJavaFilesInDirectory(sourcePath);

        // Get Sensitive variables from gpt
                    
                    // // Used for testing
                    let slice = javaFiles.slice(0, 20);  
                    const data=await this.gptService.openAiGetSensitiveVariables(slice);

        // const data = await this.gptService.openAiGetSensitiveVariables(javaFiles);

        // Replace String with findings?
        // const variablesMapping = this.formatMappings(data.sensitiveVariablesMapping);
        // const stringsMapping = this.formatMappings(data.sensitiveStringsMapping);
        // let fileContents = SensitiveVariablesContents.replace("======", data.variables.join(','));
        // fileContents = fileContents.replace("----------", variablesMapping);
        // fileContents = fileContents.replace("++++++++++", stringsMapping);



        // // Write response to file
        // await this.writeVariablesToFile(fileContents)    // commented b/c path doesn't exist
        // await this.writeFilesGptResponseToJson(data.fileList, sourcePath);  // todo



        // Remove previous database if it exists
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
        await this.fileUtilService.removeDir(db);

        // Create new database with codeql
        const createDbCommand = `database create ${db} --language=java --source-root=${sourcePath}`;
        await this.runChildProcess(createDbCommand);

        // todo try catch if failed to make db? Can run analyze if no db

        // Analyze with codeql
        const outputPath = path.join(sourcePath, 'result.sarif');
        const analyzeDbCommand = `database analyze ${db} --format=sarifv2.1.0 --output=${outputPath} ${this.queryPath}`;
        await this.runChildProcess(analyzeDbCommand);

        return await this.parserService.getSarifResults(sourcePath);

    }

    async runChatGPT(sourcePath){
        const javaFiles = await this.fileUtilService.getJavaFilesInDirectory(sourcePath);

        // Get Sensitive variables from gpt
        const data = await this.gptService.openAiGetSensitiveVariables(javaFiles);

        // Replace String with findings?
        const fileContents = this.formatMappings(data.sensitiveVariablesMapping)
        // const fileContents = SensitiveVariablesContents.replace("======", data.variables.join(','));

        // Write response to file
        await this.writeVariablesToFile(fileContents)    // commented b/c path doesn't exist
        await this.writeFilesGptResponseToJson(data.fileList, sourcePath);  // todo
    }

    async debugChatGPT(sourcePath){
        const javaFiles = await this.fileUtilService.getJavaFilesInDirectory(sourcePath);
        // Code used for testing Chat GPT Calls with preprocessing
        for(let i = 0; i < 10; i++) {
            let slice = javaFiles.slice(0, (i + 1) * 10);  
            let start = performance.now();
            const data=await this.gptService.openAiGetSensitiveVariables(slice);
            let end = performance.now();
            console.log(`Slice of ${slice.length} took ${end - start} milliseconds`);
            if(i == 0){
                fs.writeFileSync(`./times.txt`, `Slice of ${slice.length} took ${end - start} milliseconds\n`);
            }
            else{
                fs.appendFileSync(`./times.txt`, `Slice of ${slice.length} took ${end - start} milliseconds\n`);
            }

            return data;
        }
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
    async writeVariablesToFile(variables: string) {
        // todo why is this a constant path? Can be moved to config or other?
        const filePath = "../codeql/ql/java/ql/lib/semmle/code/java/security/SensitiveVariables.qll";
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

    copyQueries(srcDir: string, destDir: string) {

        fs.readdir(srcDir, { withFileTypes: true }, (err, entries) => {
            if (err) throw err;

            entries.forEach(entry => {
                const srcPath = path.join(srcDir, entry.name);
                const destPath = path.join(destDir, entry.name);

                if (entry.isDirectory()) {
                    fs.mkdir(destPath, { recursive: true }, (err) => {
                        if (err) throw err;
                        console.log(`Directory created: ${destPath}`);
                        this.copyQueries(srcPath, destPath); // Recursive call to copy directory contents
                    });
                }
            });
        });




    }

    formatMappings(mapping): string {
        let result: string = "";
        // Only consider keys with non-empty values, since this causes issues with codeql later on.
        const keys = Object.keys(mapping).filter(key => mapping[key].length > 0);
        keys.forEach((key, index) => {
            // Correctly map each variable to a string with surrounding quotes and then join them
            const variablesString = mapping[key].map(v => `${v}`).join(", ");
            result += `fileName = "${key}" and result = [${variablesString}]`;
            
            // Add the ' or\n' between entries, but not after the last entry
            if (index < keys.length - 1) {
                result += " or\n";
            }
        });
        return result;
    }
    
    
}
