import { Injectable } from '@nestjs/common';
import OpenAI from "openai";
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import { spawn } from 'child_process';

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
        this.projectPath = "";
        this.parsedResults = {};
        this.fileContents = {};
    }

    async bertWrapper(filePaths: string[], sourcePath: string) {
        this.projectPath = sourcePath;
        await this.getParsedResults(filePaths);
        await this.fileUtilService.writeToFile(path.join(this.projectPath, 'parsedResults.json'), JSON.stringify(this.parsedResults, null, 2));
        await this.getBertResponse(this.projectPath);
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

    async getBertResponse(project_root: string) {
        return new Promise((resolve, reject) => {
            const bertProcess = spawn('python', ['src/bert/run_bert.py', project_root]);

            let stdoutData = '';
            let stderrData = '';

            bertProcess.stdout.on('data', (data) => {
                stdoutData += data.toString();
                console.log(data.toString());

                // Parse progress updates
                const lines = data.toString().split('\n');
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const progressUpdate = JSON.parse(line);
                            if (progressUpdate.type && progressUpdate.progress !== undefined) {
                                this.eventsGateway.emitDataToClients(progressUpdate.type, JSON.stringify(progressUpdate));
                            }
                        } catch (e) {
                            // console.error('Failed to parse progress update:', line);
                        }
                    }
                }
            });

            bertProcess.stderr.on('data', (data) => {
                stderrData += data.toString();
            });

            bertProcess.on('close', (code) => {
                if (code === 0) {
                    resolve(stdoutData);
                } else {
                    console.warn(`Warning or error from bertProcess: ${stderrData}`);
                    resolve(null);
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
    sinks: string[];
}