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
        private fileService: FileUtilService,
        private gptService:ChatGptService
    ) {
        this.projectsPath = this.configService.get<string>(
            'CODEQL_PROJECTS_DIR',
        );

        this.queryPath = path.join(
            this.configService.get<string>('QUERY_DIR'),
            'codeql-custom-queries-java',
        );
    }
    async runCodeQl(createCodeQlDto: any) {
        var sourcePath = path.join(this.projectsPath, createCodeQlDto.project);
        var javaFiles=await this.fileService.getJavaFilesInDirectory(sourcePath);

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
        }

        const data=await this.gptService.openAiGetSensitiveVariables(javaFiles);
        const fileContents=SensitiveVariablesContents.replace("======",data.variables.join(',')) ;
        this.writeVariablesToFile(fileContents)
        this.writeFilesGptResponseToJson(data.fileList,sourcePath);
        var Db = path.join(sourcePath, createCodeQlDto.project + 'Db');
        await this.fileService.removeDir(Db);
        var createDbCommand = `database create ${Db} --language=java --source-root=${sourcePath}`;
        await this.runChildProcess(createDbCommand);
        var outputPath = path.join(sourcePath, 'result.sarif');
        var analyzeDbCommand = `database analyze ${Db} --format=sarifv2.1.0 --output=${outputPath} ${this.queryPath}`;
        await this.runChildProcess(analyzeDbCommand);
        
        return await this.parserService.getSarifResults(sourcePath);
        
    }

    runChildProcess(codeQlCommand: string): Promise<void> {
        var commands = codeQlCommand.split(' ');
        return new Promise((resolve, reject) => {
            let childProcess = spawn('codeql', commands);
            childProcess.stdout.on('data', (data) => {
                console.log(data.toString());
                this.eventsGateway.emitDataToClients('data',data.toString())
            });
            childProcess.stderr.on('data', (data) => {
                console.log(data.toString());
                this.eventsGateway.emitDataToClients('data',data.toString())
            });
            const self = this;
            childProcess.on('exit', function (code, signal) {
                const result = "process CodeQl exited with code "+code+" and signal "+signal;
                console.log(result);
                self.eventsGateway.emitDataToClients('data',result.toString())
                resolve();
            });

            childProcess.on('error', (error) => {
                console.log(error);
                reject(error);
            });
        });
    }

    async writeVariablesToFile(variables:string){
        var filePath="../codeql/ql/java/ql/lib/semmle/code/java/security/SensitiveVariables.qll";
        await this.fileService.writeToFile(filePath,variables) 
    }

    async writeFilesGptResponseToJson(fileList:any[],sourcePath:string){
        var jsonPath=path.join(sourcePath,"data.json");
        var data=JSON.stringify(fileList);
        await this.fileService.writeToFile(jsonPath,data)
    }
}
