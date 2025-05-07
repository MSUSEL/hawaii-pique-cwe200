import { Global, Injectable } from '@nestjs/common';

import * as path from 'path';
import * as fs from 'fs';
import { exec} from 'child_process';
import { ConfigService } from '@nestjs/config';
import { EventsGateway } from 'src/events/events.gateway';
import { FileUtilService } from 'src/files/fileUtilService';

/**
 * JavaParserService is a service that provides functionality to parse Java files and extract relevant information.
 */
@Global()
@Injectable()
export class JavaParserService {
    projectsPath:string;
    parsedResults: { [key: string]: JavaParseResult };
    projectPath: string;

    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
        ){
            this.projectsPath=this.configService.get<string>('CODEQL_PROJECTS_DIR')
            
        }
    /**
     * A wrapper that kicks off the parsing.
     * @param filePaths - An array of file paths to be parsed.
     * @param sourcePath - The path to the source directory.
     * @returns A promise that resolves to an object containing the parsed results.
     * @throws Error if the parsing fails.
     * */
    async wrapper(filePaths: string[], sourcePath: string) {
        this.projectPath = sourcePath;
        await this.buildJarIfNeeded();
        this.parsedResults = await this.getParsedResults(filePaths);
        await this.fileUtilService.writeToFile(path.join(this.projectPath, 'parsedResults.json'), JSON.stringify(this.parsedResults, null, 2));
    }
    
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
            const results: { [key: string]: JavaParseResult } = {};
        
            const workers = Array.from({ length: concurrencyLimit }, async () => {
                while (queue.length > 0) {
                    const filePath = queue.shift();
                    if (!filePath) break;
        
                    const result = await this.parseJavaFile(filePath);
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

        /**
         * Builds the "ParseJava" JAR file if it doesn't exist.
         * @returns A promise that resolves when the JAR file is built.
         **/
        async buildJarIfNeeded() {
                const cwd = process.cwd();
                const jarPath = path.resolve(cwd, 'ParseJava', 'target', 'ParseJava-1.0-jar-with-dependencies.jar');
            
                if (!fs.existsSync(jarPath)) {
                    // Build the JAR file
                    console.log("Building JAR for Java Parser")
                    await new Promise<void>((resolve, reject) => {
                        exec('mvn clean package', { cwd: path.resolve(cwd, 'ParseJava') }, (error, stdout, stderr) => {
                            if (error) {
                                reject(`Error building JAR: ${stderr}`);
                                return;
                            }
                            resolve();
                        });
                    });
                }
            }
            
            /**
             * Wrapper for calling the JAR in "backend/ParseJava/target/ParseJava-1.0-jar-with-dependencies.jar".
             * @param filePath - The path to the Java file to be parsed.
             * @returns A promise that resolves to an object containing the parsed results.
             **/
            async parseJavaFile(filePath: string): Promise<JavaParseResult> {
                const cwd = process.cwd();
                const jarPath = path.resolve(cwd, 'ParseJava', 'target', 'ParseJava-1.0-jar-with-dependencies.jar');
                filePath = path.resolve(cwd, filePath);
            
                // Run the Java program
                const result = await this.runJavaProgram(jarPath, filePath);
                return result;
            }
            
            
            /**
             * Executes the actual Java program using the JAR file and the provided file path which is the source code to be parsed.
             * @param jarPath 
             * @param filePath 
             * @returns A promise that resolves to the parsed result.
             */
            async runJavaProgram(jarPath: string, filePath: string): Promise<JavaParseResult> {
                return new Promise((resolve, reject) => {
                    const command = `java -jar ${jarPath} ${filePath}`;
            
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
                                methodCodeMap:[]
                            });
                            return;
                        }
            
                        try {
                            const result: JavaParseResult = JSON.parse(stdout);
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
    interface JavaParseResult {
        filename: string;
        variables: string[];
        comments: string[];
        strings: string[];
        sinks: string[];
        methodCodeMap: string[];
    }