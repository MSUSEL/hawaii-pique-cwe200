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
    projectPath: string;
    encode: any;
    parsedResults:{ [key: string]: JavaParseResult };
    fileContents = {};
    
    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
    ) {

        this.projectPath = ""
        this.parsedResults = {};
        this.fileContents = {};
    }

    async bertWrapper(filePaths: string[], sourcePath: string){
        this.projectPath = sourcePath
        await this.getParsedResults(filePaths);
        await this.fileUtilService.writeToFile(path.join(this.projectPath, 'parsedResults.json'), JSON.stringify(this.parsedResults, null, 2));
        await this.getBertResponse(this.projectPath);

    }

    async getParsedResults(filePaths: string[]){
        for (let filePath of filePaths){
            await this.fileUtilService.parseJavaFile(filePath, this.parsedResults);
        }
    
    }

    async getBertResponse(project_root) {
        return new Promise((resolve, reject) => {
            const bertProcess = spawn('python', ['src/bert/run_bert.py', project_root]);
    
            let stdoutData = '';
            let stderrData = '';
    
            bertProcess.stdout.on('data', (data) => {
                stdoutData += data.toString();
            });
    
            bertProcess.stderr.on('data', (data) => {
                stderrData += data.toString();
            });
    
            bertProcess.on('close', (code) => {
                if (code === 0) {
                    resolve(stdoutData);
                } else {
                    // Only print the error if it's not the specific warning we're ignoring
                    // if (!stderrData.includes('tf.losses.sparse_softmax_cross_entropy') && !stderrData.includes('Error in loading the saved optimizer state')) {
                        console.warn(`Warning or error from bertProcess: ${stderrData}`);
                    // }
                    resolve(null); // Resolve with null to indicate an issue, but avoid breaking the flow
                }
            });
    
            bertProcess.on('error', (err) => {
                reject(`Failed to start subprocess: ${err}`);
            });
        });
    }
    
}

interface JavaParseResult {
    filename: string;
    variables: string[];
    comments: string[];
    strings: string[];
}
