import { Injectable } from '@nestjs/common';
import OpenAI from "openai";
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import async from 'async';
import { spawn } from 'child_process';
// import {VariableParser, StringParser, CommentParser, SinkParser} from './JSON-parsers'
import { json } from 'stream/consumers';

@Injectable()
export class BertService {
    projectsPath: string;
    encode: any;
    parsedResults:{ [key: string]: JavaParseResult };
    fileContents = {};
    
    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
    ) {

        this.projectsPath = this.configService.get<string>('CODEQL_PROJECTS_DIR',);
        this.parsedResults = {};
        this.fileContents = {};
    }

    async bertWrapper(filePaths: string[]){
        await this.getParsedResults(filePaths);
        await this.readFiles(filePaths);
        await this.getBertResponse(this.fileContents, this.parsedResults);

    }

    async getParsedResults(filePaths: string[]){
        for (let filePath of filePaths){
            await this.fileUtilService.parseJavaFile(filePath, this.parsedResults);
        }
    
    }

    async readFiles(filePaths: string[]){
        for (let filePath of filePaths){
            let baseName = path.basename(filePath);
            this.fileContents[baseName] = this.fileUtilService.processJavaFile(filePath);
        }
    }

    async getBertResponse(fileContents, parsedValues){ {
        return new Promise((resolve, reject) => {
            const bertProcess = spawn('python', ['src/bert/bert.py', "filePath"]);
            bertProcess.stdout.on('data', (data) => {
                resolve(data.toString());
            });
            bertProcess.stderr.on('data', (data) => {
                reject(data.toString());
            });
        });
    }}
}

interface JavaParseResult {
    filename: string;
    variables: string[];
    comments: string[];
    strings: string[];
}
