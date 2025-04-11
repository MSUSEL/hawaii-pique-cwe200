import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import { CodeQlParserService } from '../code-ql/codql-parser-service';

import { ConfigService } from '@nestjs/config';
import { CodeQlService } from 'src/code-ql/code-ql.service';
import { BertService } from 'src/bert/bert.service';
import { LLMService } from 'src/llm/llm.service';

import { SensitiveVariables } from 'src/utils/SensitiveVariables';
import { SensitiveComments } from 'src/utils/SensitiveComments';
import { SensitiveStrings } from 'src/utils/SensitiveStrings';
import { Sinks } from 'src/utils/Sinks';

import * as path from 'path';

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
     *
     * @param createAnalyzeDto Data transfer object with project name, ?java version, and ?extension
     */
    async runAnalysis(createAnalyzeDto: any) {
        // Check if a java version is specified
        // if (createAnalyzeDto.javaVersion) {
        //     this.fileUtilService.setJavaVersion(Number(createCodeQlDto.javaVersion));
        // }

        // Get all java files in project
        const sourcePath = path.join(this.projectsPath, createAnalyzeDto.project);
        const javaFiles = await this.fileUtilService.getJavaFilesInDirectory(sourcePath);

        // await this.executeStep('Parsing files for variables, strings, comments, and method calls.', async () => {
        //     await this.bertService.bertWrapper(javaFiles, sourcePath);
        // });

        // await this.executeStep('Detecting sensitive info using BERT.', async () => {
        //     await this.bertService.getBertResponse(sourcePath, "run_bert.py");
        // });
        
        // let data = null;
        // await this.executeStep('Reading in BERT results from data.json.', async () => {
        //     data = this.fileUtilService.parseJSONFile(path.join(sourcePath, 'data.json'));
        // });

        // await this.executeStep('Saving the sensitive info to .yml files.', async () => {
        //     await this.saveSensitiveInfo(data);
        // });

        // await this.executeStep('Creating CodeQL database.', async () => {
        //     await this.codeqlService.createDatabase(sourcePath, createAnalyzeDto);
        // });

        // await executeStep('Running the backward slice queries.', async () => {
        //     await this.performBackwardSlicing(sourcePath, createCodeQlDto);
        // });

        // await executeStep('Parsing the backward slice graphs.', async () => {
        //     await this.bertService.parseBackwardSlice(sourcePath);
        // });

        // await executeStep('Running BERT with backward slice graphs.', async () => {
        //     await this.bertService.getBertResponse(sourcePath, 'bert_with_graph.py');
        // });

        // await executeStep('Updating sensitiveVariables.yml.', async () => {
        //     const sensitiveVariables = this.useSavedData(sourcePath, 'sensitiveVariables.json');
        //     this.saveUpdatedSensitiveVariables(sensitiveVariables);
        // });

        // await this.executeStep('Running CWE queries.', async () => {
        //     await this.codeqlService.runCWEQueries(sourcePath, createAnalyzeDto);
        // });

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


    // Helper function to format time in minutes and seconds
    formatTime = (seconds: number) => {
        if (seconds > 60) {
            const minutes = Math.floor(seconds / 60); // Get the whole minutes
            const remainingSeconds = Math.floor(seconds % 60); // Get the remaining seconds
            return `${minutes} minute${minutes > 1 ? 's' : ''} and ${remainingSeconds} second${remainingSeconds > 1 ? 's' : ''}`;
        } else {
            return `${Math.floor(seconds)} second${seconds > 1 ? 's' : ''}`; // Show seconds
        }
    };

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

    // Helper function to record the time taken for each step
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

    async saveSensitiveInfo(data) {
        const variablesMapping = this.bertService.formatMappings(data.sensitiveVariablesMapping, "variables");
        const stringsMapping = this.bertService.formatMappings(data.sensitiveStringsMapping, "strings");
        const commentsMapping = this.bertService.formatCommentsMapping(data.sensitiveCommentsMapping, "comments");
        const sinksMapping = this.bertService.formatSinkMappings(data.sinksMapping);
        
        const codeQLQueriesPath = path.join("..", "codeql queries", "SensitiveInfo") // Path to codeql queries dir
        const qlPath = path.join(this.queryPath, "SensitiveInfo") // Path to codeql dir

        let variablesFile = SensitiveVariables.replace("----------", variablesMapping);
        await this.fileUtilService.writeToFile(path.join(codeQLQueriesPath, "SensitiveVariables.yml"), variablesFile, )
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

    async saveUpdatedSensitiveVariables(data) {
        const variablesMapping = this.bertService.formatMappings(data.sensitiveVariablesMapping, "variables");
        let variablesFile = SensitiveVariables.replace("----------", variablesMapping);
        await this.fileUtilService.writeToFile(variablesFile, "../codeql queries/SensitiveInfo/SensitiveVariables.yml")
        await this.fileUtilService.writeToFile(variablesFile, "../codeql/codeql-custom-queries-java/SensitiveInfo/SensitiveVariables.yml")

    }
}

