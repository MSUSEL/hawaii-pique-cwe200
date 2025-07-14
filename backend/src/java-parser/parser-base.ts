import * as path from 'path';
import * as fs from 'fs';
import { exec } from 'child_process';
import { EventsGateway } from 'dist/src/events/events.gateway';


export abstract class ParserBase {
    projectsPath:string;
    parsedResults: { [key: string]: ParseResult };
    projectPath: string;

    protected eventsGateway: EventsGateway;

    constructor(eventsGateway: EventsGateway) {
        this.eventsGateway = eventsGateway;
    }


    abstract wrapper(filePaths: string[], sourcePath: string);

    /**
     * Used to call the parser program.
     * For example, 
     * @param filePath 
     */
    abstract parseSourceFile(filePath: string): Promise<ParseResult>;





    /**
     * Parses the given Java files and extracts relevant information.
     * @param filePaths - An array of file paths to be parsed.
     * @returns A promise that resolves to an object containing the parsed results.
     **/
    async getParsedResults(filePaths: string[]) {
            const total = filePaths.length;
            let completed = 0;
            let previousProgress = 0;
        
            // Limit the number of concurrent tasks
            const concurrencyLimit = 8; // Adjust based on your system's capabilities
            const queue = filePaths.slice(); // Create a copy of the file paths
            const results: { [key: string]: ParseResult } = {};
            const parserProgramPath = path.join(this.projectsPath, 'ParseJava', 'target', 'ParseJava-1.0-jar-with-dependencies.jar');
        
            const workers = Array.from({ length: concurrencyLimit }, async () => {
                while (queue.length > 0) {
                    const filePath = queue.shift();
                    if (!filePath) break;
        
                    const result = await this.parseSourceFile(filePath);
                    results[path.basename(result.filename)] = result;
        
                    completed += 1;
                    
                    const progressPercent = Math.floor((completed / total) * 100);
                    if (progressPercent !== previousProgress) {
                        previousProgress = progressPercent;
                        console.log(`Parsing progress: ${progressPercent}%`);
                        this.eventsGateway.emitDataToClients('parsingProgress', JSON.stringify({ type: 'parsingProgress', parsingProgress: progressPercent }));
                    }
                    // console.log(`Parsing progress: ${progressPercent}%`);
                    // this.eventsGateway.emitDataToClients('parsingProgress', JSON.stringify({ type: 'parsingProgress', parsingProgress: progressPercent }));
                }
            });
        
            await Promise.all(workers);
        
            // Assign the aggregated results
            return results;
        }


    async runExternalParser(command: string, filePath: string): Promise<ParseResult> {
        return new Promise((resolve, reject) => {

            exec(command, (error, stdout, stderr) => {
                if (error) {
                    console.error(`Error executing command for file ${filePath}: ${stderr}`);
                    // Return an empty result for this file
                    resolve({
                        filename: path.basename(filePath),
                        variables: [],
                        comments: [],
                        strings: [],
                        sinks: [],
                        methodCodeMap: []
                    });
                    return;
                }

                try {
                    const result: ParseResult = JSON.parse(stdout);
                    resolve(result);
                } catch (e) {
                    // console.error(`Failed to parse JSON: ${e}`);
                    resolve({
                        filename: path.basename(filePath),
                        variables: [],
                        comments: [],
                        strings: [],
                        sinks: [],
                        methodCodeMap: []
                    });
                }
            });
        });
    }
}


/**
 * Interface representing the result of parsing a Java file.
 * @property {string} filename - The name of the Java file.
 * @property {string[]} variables - An array of variable names found in the file.
 * @property {string[]} comments - An array of comments found in the file.
 * @property {string[]} strings - An array of string literals found in the file.
 * @property {string[]} sinks - An array of sink points found in the file.
 * @property {string[]} methodCodeMap - An array of method code mappings found in the file defines context for the Attack Surface Detection Engine.
 * 
 */
export interface ParseResult {
    filename: string;
    variables: string[];
    comments: string[];
    strings: string[];
    sinks: string[];
    methodCodeMap: string[];
}