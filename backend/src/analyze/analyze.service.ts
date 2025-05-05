import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import { CodeQlParserService } from '../code-ql/codql-parser-service';

import { ConfigService } from '@nestjs/config';
import { CodeQlService } from 'src/code-ql/code-ql.service';
import { BertService } from 'src/bert/bert.service';
import { LLMService } from 'src/llm/llm.service';
import { JavaParserService } from 'src/java-parser/java-parser.service';

import { SensitiveVariables } from 'src/utils/SensitiveVariables';
import { SensitiveComments } from 'src/utils/SensitiveComments';
import { SensitiveStrings } from 'src/utils/SensitiveStrings';
import { Sinks } from 'src/utils/Sinks';

import * as path from 'path';

/**
 * Service responsible for analyzing Java projects, detecting sensitive information,
 * and verifying data flows using various engines and tools.
 */
@Injectable()
export class AnalyzeService {
    projectsPath: string;
    queryPath: string;
    private times: Record<string, number> = {};

    constructor(
        private configService: ConfigService,
        private parserService: CodeQlParserService,
        private fileUtilService: FileUtilService,
        private bertService: BertService,
        private llmService: LLMService,
        private codeqlService: CodeQlService,
        private javaParserService: JavaParserService,
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
     * Run analysis on a project
     * This is the main function that kicks off the analysis process.
     * It performs the following steps:
     * 1. Set the Java version if specified
     * 2. Get all Java files in the project directory
     * 3. [Attack Surface Detection Engine] Parse the files for variables, strings, comments, and method calls
     * 4. [Attack Surface Detection Engine] Detect sensitive information using BERT
     * 5. [Attack Surface Detection Engine] Read in BERT results from data.json
     * 6. [Attack Surface Detection Engine] Save the sensitive information to .yml files
     * 7. [Exposure Analysis Engine] Create a CodeQL database
     * 8. [Exposure Analysis Engine] Run CWE queries
     * 9. [Exposure Analysis Engine] Save the Dataflow Tree
     * 10. [Flow Verification Engine]Verify Data Flows if any flows are detected
     * 11. [Flow Verification Engine] Update the SARIF file
     * 12. Print the time taken for each step
     * 13. Return the results in the specified format (CSV or SARIF)
     * @param createAnalyzeDto Data transfer object with project name, ?java version, and ?extension
     */
    async runAnalysis(createAnalyzeDto: any) {
        // Check if a java version is specified
        if (createAnalyzeDto.javaVersion) {
            this.fileUtilService.setJavaVersion(Number(createAnalyzeDto.javaVersion));
        }

        // Get all java files in project
        const sourcePath = path.join(this.projectsPath, createAnalyzeDto.project);
        const javaFiles = await this.fileUtilService.getJavaFilesInDirectory(sourcePath);

        await this.executeStep('Parsing files for variables, strings, comments, and method calls.', async () => {
            await this.javaParserService.wrapper(javaFiles, sourcePath);
        });

        await this.executeStep('Detecting sensitive info using BERT.', async () => {
            await this.bertService.getBertResponse(sourcePath, "run_bert.py");
        });

        let data = null;
        await this.executeStep('Reading in BERT results from data.json.', async () => {
            data = this.fileUtilService.parseJSONFile(path.join(sourcePath, 'data.json'));
        });

        await this.executeStep('Saving the sensitive info to .yml files.', async () => {
            await this.saveSensitiveInfo(data);
        });

        await this.executeStep('Creating CodeQL database.', async () => {
            await this.codeqlService.createDatabase(sourcePath, createAnalyzeDto);
        });

        await this.executeStep('Running CWE queries.', async () => {
            await this.codeqlService.runCWEQueries(sourcePath, createAnalyzeDto);
        });

        await this.executeStep('Saving Dataflow Tree.', async () => {
            await this.parserService.saveDataFlowTree(sourcePath);
        });

        const flows = await this.fileUtilService.readJsonFile(path.join(sourcePath, 'flowMapsByCWE.json'));
        if (Object.keys(flows).length > 0) {

            await this.executeStep('Verifying Data Flows.', async () => {
                await this.bertService.verifyFlows(sourcePath);
            });

            await this.executeStep('Updating Sarif.', async () => {
                await this.codeqlService.updateSarif(sourcePath);
            });
        }

        // Print all the times at the end
        console.log("Time taken for each step:");
        Object.keys(this.times).forEach(step => {
            console.log(`${step}: ${this.formatTime(this.times[step])}`);
        });

        if (createAnalyzeDto.extension === 'csv') {
            return await this.parserService.getcsvResults(sourcePath);
        }

        return await this.parserService.getSarifResults(sourcePath);
    }

    /**
     * Formats the time taken for a step in the analysis process.
     * Converts seconds into a human-readable format (e.g., "2 minutes and 30 seconds").
     * 
     * @param seconds - The time taken in seconds.
     * @returns A formatted string representing the time.
     */
    formatTime = (seconds: number) => {
        if (seconds > 60) {
            const minutes = Math.floor(seconds / 60); // Get the whole minutes
            const remainingSeconds = Math.floor(seconds % 60); // Get the remaining seconds
            return `${minutes} minute${minutes > 1 ? 's' : ''} and ${remainingSeconds} second${remainingSeconds > 1 ? 's' : ''}`;
        } else {
            return `${Math.floor(seconds)} second${seconds > 1 ? 's' : ''}`; // Show seconds
        }
    };

    /**
     * Executes a specific step in the analysis process and records the time taken.
     * Logs the step name and handles any errors that occur during execution.
     * 
     * @param stepName - The name of the step being executed.
     * @param stepFunction - The function to execute for the step.
     */
    async executeStep(stepName, stepFunction) {
        console.log('-----------------------------------')
        console.log(stepName)
        console.log('-----------------------------------')


        try {
            await this.recordTime(stepName, stepFunction);
        } catch (error) {
            console.error(`Error in ${stepName}:`, error);
            // Notify the client about the error
            console.log(`Error in ${stepName}: ${error.message}`);
            // Stop further execution by re-throwing the error
            throw error(error.message);
        }
    }

    /**
     * Records the time taken for a specific function to execute.
     * 
     * @param stepName - The name of the step being recorded.
     * @param fn - The function to execute and measure.
     */
    recordTime = async (stepName: string, fn: () => Promise<void>) => {
        const start = Date.now();
        await fn();
        const end = Date.now();
        this.times[stepName] = (end - start) / 1000; // Time in seconds
    };


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


    /**
     * Saves sensitive information mappings (variables, strings, comments, and sinks) to YAML files.
     * These files are used for CodeQL queries.
     * 
     * @param data - The data containing sensitive information mappings.
     */
    async saveSensitiveInfo(data) {
        const variablesMapping = this.bertService.formatMappings(data.sensitiveVariablesMapping, "variables");
        const stringsMapping = this.bertService.formatMappings(data.sensitiveStringsMapping, "strings");
        const commentsMapping = this.bertService.formatCommentsMapping(data.sensitiveCommentsMapping, "comments");
        const sinksMapping = this.bertService.formatSinkMappings(data.sinksMapping);

        const codeQLQueriesPath = path.join("..", "codeql queries", "SensitiveInfo") // Path to codeql queries dir
        const qlPath = path.join(this.queryPath, "SensitiveInfo") // Path to codeql dir

        let variablesFile = SensitiveVariables.replace("----------", variablesMapping);
        await this.fileUtilService.writeToFile(path.join(codeQLQueriesPath, "SensitiveVariables.yml"), variablesFile,)
        await this.fileUtilService.writeToFile(path.join(qlPath, "SensitiveVariables.yml"), variablesFile)

        let stringsFile = SensitiveStrings.replace("++++++++++", stringsMapping);
        await this.fileUtilService.writeToFile(path.join(codeQLQueriesPath, "SensitiveStrings.yml"), stringsFile)
        await this.fileUtilService.writeToFile(path.join(qlPath, "SensitiveStrings.yml"), stringsFile)

        let commentsFile = SensitiveComments.replace("**********", commentsMapping);
        await this.fileUtilService.writeToFile(path.join(codeQLQueriesPath, "SensitiveComments.yml"), commentsFile)
        await this.fileUtilService.writeToFile(path.join(qlPath, "SensitiveComments.yml"), commentsFile)

        let sinksFile = Sinks.replace("----------", sinksMapping);
        await this.fileUtilService.writeToFile(path.join(codeQLQueriesPath, "Sinks.yml"), sinksFile)
        await this.fileUtilService.writeToFile(path.join(qlPath, "Sinks.yml"), sinksFile)

    }

    /**
     * [Unused] Was used to update the variables when backslicing was used.
     * Updates and saves sensitive variable mappings to YAML files.
     * 
     * @param data - The data containing updated sensitive variable mappings.
     */
    async saveUpdatedSensitiveVariables(data) {
        const variablesMapping = this.bertService.formatMappings(data.sensitiveVariablesMapping, "variables");
        let variablesFile = SensitiveVariables.replace("----------", variablesMapping);
        await this.fileUtilService.writeToFile(variablesFile, "../codeql queries/SensitiveInfo/SensitiveVariables.yml")
        await this.fileUtilService.writeToFile(variablesFile, "../codeql/codeql-custom-queries-java/SensitiveInfo/SensitiveVariables.yml")

    }
}

