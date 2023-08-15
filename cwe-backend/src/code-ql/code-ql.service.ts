import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import { exec, spawn } from 'child_process';
import { CodeQlParserService } from './codql-parser-service';
import * as path from 'path';
import * as fs from 'fs';
import { ConfigService } from '@nestjs/config';
import { ChatGptService } from 'src/chat-gpt/chat-gpt.service';
import { SensitiveVariablesContents } from './data';
@Injectable()
export class CodeQlService {
    projectsPath: string;
    queryPath: string;
    constructor(
        private configService: ConfigService,
        private parserService: CodeQlParserService,
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
        
        const variables=await this.gptService.openAiGetSensitiveVariables(javaFiles);
        const fileContents=SensitiveVariablesContents.replace("======",variables.join(',')) ;
        this.writeVariablesToFile(fileContents)
       
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
            });
            childProcess.stderr.on('data', function (data) {
                console.log(data.toString());
            });

            childProcess.on('exit', function (code, signal) {
                const result = `process CodeQl exited with code ${code} and signal ${signal}`;
                console.log(result);
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
}
