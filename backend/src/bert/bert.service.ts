import { Injectable } from '@nestjs/common';
import OpenAI from "openai";
import { ConfigService } from '@nestjs/config';
import { FileUtilService } from 'src/files/fileUtilService';
import { EventsGateway } from 'src/events/events.gateway';
import * as path from 'path';
import { spawn } from 'child_process';
import { JavaParserService } from 'src/java-parser/implementations/java-parser.service';

/**
 * Service responsible for handling BERT-related operations, including parsing,
 * backward slicing, flow verification, and formatting mappings.
 */
@Injectable()
export class BertService {
    projectPath: string;
    encode: any;
    parsedResults: { [key: string]: JavaParseResult };
    fileContents = {};
    contextMap = {};

    constructor(
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
        private javaParserService: JavaParserService,
    ) { }

    /**
     * Executes a Python script to get a response from a BERT model.
     * Emits progress updates to connected clients via the EventsGateway.
     * 
     * @param project_root - The root directory of the project.
     * @param bertScript - The name of the Python script to execute.
     * @returns A promise that resolves with the script's standard output or null if an error occurs.
     */

    async getBertResponse(project_root: string, bertScript: string) {
        return new Promise((resolve, reject) => {
            const bertProcess = spawn('python', [path.join("src", "bert", "inference", bertScript), project_root]);

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
            bertProcess.stderr.on('data', (data) => {
                stderrData += data.toString();
                console.error('Python script error output:', data.toString());  // Print script's error output
            });

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

    /**
     * [Unused] Calls a Python script to parse backward slices from a SARIF file.
     * 
     * @param sourcePath - The root path of the source project.
     * @returns A promise that resolves when the process completes successfully.
     */
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

    /**
     * FlowVerificationEngine
     * Calls a Python script to verify flow maps that are stored in the project's flowMapsByCWE.json file.
     * 
     * @param sourcePath - The root path of the source project.
     * @returns A promise that resolves when the process completes successfully.
     */
    async verifyFlows(sourcePath: string) {
        const jsonFilePath = path.join(sourcePath, 'flowMapsByCWE.json');
        const projectName = path.basename(sourcePath);

        return new Promise<void>((resolve, reject) => {
            // Spawn the Python process
            const process = spawn('python', [path.join("src", "bert", "inference", "flow_verification.py"), projectName]);
            console.log(`Running verification script on ${projectName}`);

            // Capture stderr data
            process.stderr.on('data', (data) => {
                console.error(`stderr: ${data}`);
            });

            // Handle the process exit
            process.on('close', (code) => {
                if (code === 0) {
                    console.log('Flows successfully verified.');
                    resolve();
                } else {
                    console.error(`Flow verification process exited with code ${code}`);
                    reject(new Error(`Flow verification process exited with code ${code}`));
                }
            });
        });
    }

    /**
     * Formats a mapping of filenames to variables into a specific string format.
     * This done so that the output can be used in YAML files for CodeQL queries.
     * 
     * @param mapping - An object where keys are filenames and values are arrays of strings.
     * @param type - The type of data being formatted (e.g., "variables").
     * @returns A formatted string representation of the mapping.
     */
    formatMappings(mapping: { [key: string]: string[] }, type: string): string {
        let result = "";

        // Iterate over each key (filename) in the mapping object
        Object.keys(mapping).forEach(key => {
            // For each variable associated with the key, generate a new line in the output
            mapping[key].forEach(variable => {
                // Remove Unicode characters, double quotes, single quotes, and newlines from the variable
                // Ensure backslashes are escaped
                variable = variable.replace(/[^\x00-\x7F]/g, '').replace(/["'\n]/g, '').replace(/\\/g, '\\\\');

                // Apply exclusion rules
                if (
                    // Exclude common non-sensitive patterns
                    /example/i.test(variable) ||
                    /test/i.test(variable) ||
                    /demo/i.test(variable) ||
                    /foo/i.test(variable) ||
                    /bar/i.test(variable) ||
                    /baz/i.test(variable) ||
                    /secret/i.test(variable) ||
                    // Exclude empty strings
                    variable === "" ||
                    // Exclude whitespace-only strings
                    /^\s*$/.test(variable) ||
                    // Exclude strings with exactly one dot followed by a digit
                    /^[^.]*\.[0-9]+$/.test(variable)
                ) {
                    return; // Skip this variable if it matches any exclusion criteria
                }

                // Check if the variable is in the format {key: value, key: value}
                const regex = /{([^}]+)}/;
                const match = regex.exec(variable);
                if (match) {
                    // Extract values from the variable
                    const valuesString = match[1];
                    const values = valuesString.split(',').map(pair => {
                        const parts = pair.split(':');
                        return parts.length > 1 ? parts[1].trim() : '';
                    });
                    if (type === "variables") {
                        values.forEach(value => {
                            result += `    - ["${key}", ${value}]\n`;
                        });
                    } else {
                        values.forEach(value => {
                            result += `    - ["${key}", "${value}"]\n`;
                        });
                    }
                } else {
                    // If not in the format {key: value, key: value}, handle as normal string
                    if (type === "variables") {
                        result += `    - ["${key}", "${variable}"]\n`;
                    } else {
                        result += `    - ["${key}", "${variable}"]\n`;
                    }
                }
            });
        });

        return result;
    }

    /**
     * Formats a mapping of comments into a specific string format.
     * This done so that the output can be used in YAML files for CodeQL queries.
     * 
     * @param mapping - An object where keys are filenames and values are arrays of strings.
     * @param type - The type of data being formatted (e.g., "comments").
     * @returns A formatted string representation of the mapping.
     */
    formatCommentsMapping(mapping: { [key: string]: string[] }, type: string): string {
        let result = "";

        Object.keys(mapping).forEach(key => {
            mapping[key].forEach(variable => {
                let cleanValue = variable.trim();

                // Remove surrounding quotes if present
                if (cleanValue.startsWith('"') && cleanValue.endsWith('"')) {
                    cleanValue = cleanValue.slice(1, -1);
                }

                // Flatten multiline content to a single line, escape inner quotes
                const safeValue = cleanValue
                    .replace(/\r?\n\s*/g, ' ')  // Replace newlines + indentation with space
                    .replace(/"/g, '\\"');      // Escape double quotes inside

                result += `    - ["${key}", "${safeValue}"]\n`;
            });
        });

        return result;
    }


    /**
     * Formats a mapping of sinks into a specific string format.
     * This done so that the output can be used in YAML files for CodeQL queries.
     * 
     * @param mapping - An object where keys are filenames and values are arrays of arrays of strings.
     * @returns A formatted string representation of the mapping.
     */
    formatSinkMappings(mapping: Map<string, string[][]>): string {
        let result = "";
        // Iterate over each key-value pair in the mapping object
        for (const [key, sinks] of mapping.entries()) {
            // For each array associated with the key, generate a new line in the output
            for (const sink of sinks) {
                result += `    - ["${key}", "${sink[0]}", "${sink[1]}"]\n`;
            }
        }
        return result;
    }

    /**
     * Formats an array of strings into a specific string format.
     * This done so that the output can be used in YAML files for CodeQL queries.
     * 
     * @param inputArray - An array of strings to be formatted.
     * @returns A formatted string representation of the array.
     **/

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
}



interface JavaParseResult {
    filename: string;
    variables: string[];
    comments: string[];
    strings: string[];
    sinks: string[];
}
