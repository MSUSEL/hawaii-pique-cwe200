import { Injectable } from '@nestjs/common';
import OpenAI from "openai";
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import { spawn } from 'child_process';
import { JavaParserService } from 'src/java-parser/java-parser.service';

@Injectable()
export class BertService {
    projectPath: string;
    encode: any;
    parsedResults: { [key: string]: JavaParseResult };
    fileContents = {};
    contextMap = {};

    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
        private javaParserService: JavaParserService,
    ) {} 

    async bertWrapper(filePaths: string[], sourcePath: string) {
        this.projectPath = sourcePath;
        await this.fileUtilService.buildJarIfNeeded();
        this.parsedResults = await this.javaParserService.getParsedResults(filePaths);
        await this.fileUtilService.writeToFile(path.join(this.projectPath, 'parsedResults.json'), JSON.stringify(this.parsedResults, null, 2));
    }

    async getBertResponse(project_root: string, bertScript: string) {
        return new Promise((resolve, reject) => {
            const bertProcess = spawn('python', [path.join("src", "bert", bertScript), project_root]);
    
            let stdoutData = '';
            let stderrData = '';
    
            // Handle standard output (stdout)
            bertProcess.stdout.on('data', (data) => {
                stdoutData += data.toString();
                console.log(data.toString());  // Print script's output
    
                const lines = data.toString().split('\n');
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            // Only try to parse if the line starts with '{' and ends with '}'
                            if (line.trim().startsWith('{') && line.trim().endsWith('}')) {
                                const progressUpdate = JSON.parse(line);
                                if (progressUpdate.type && progressUpdate.progress !== undefined) {
                                    this.eventsGateway.emitDataToClients(progressUpdate.type, JSON.stringify(progressUpdate));
                                }
                            // } else {
                            //     console.log('Non-JSON output:', line); // Log non-JSON lines like progress bars
                            }
                        } catch (e) {
                            console.error('Failed to parse progress update:', line);  // Log parse errors
                        }
                    }
                }
            });
    
            // Handle standard error (stderr)
            // bertProcess.stderr.on('data', (data) => {
            //     stderrData += data.toString();
            //     console.error('Python script error output:', data.toString());  // Print script's error output
            // });
    
            // Handle when the process closes
            bertProcess.on('close', (code) => {
                if (code === 0) {
                    console.log('Python script finished successfully');
                    resolve(stdoutData);
                } else {
                    console.warn(`Warning or error from bertProcess: ${stderrData}`);
                    resolve(null);
                }
            });
    
            // Handle process errors (e.g., failure to start)
            bertProcess.on('error', (err) => {
                reject(`Failed to start subprocess: ${err}`);
            });
        });
    }
    

    async parseBackwardSlice(sourcePath: string) {
        const inputPath = path.join(sourcePath, 'backwardslice.sarif');
        const outputPath = path.join(sourcePath, 'variables_graph.json');
        
        // Parse the results to create the backslice graph for each variable
        return new Promise<void>((resolve, reject) => {
            // Spawn the Python process
            const backsliceProcess = spawn('python', [path.join("src", "bert", "backslicing.py"), inputPath, outputPath]);
    
            // Capture stderr data
            backsliceProcess.stderr.on('data', (data) => {
                console.error(`stderr: ${data}`);
            });
    
            // Handle the process exit
            backsliceProcess.on('close', (code) => {
                if (code === 0) {
                    console.log('backslicing.py executed successfully.');
                    resolve();
                } else {
                    console.error(`backslicing.py process exited with code ${code}`);
                    reject(new Error(`backslicing.py process exited with code ${code}`));
                }
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
