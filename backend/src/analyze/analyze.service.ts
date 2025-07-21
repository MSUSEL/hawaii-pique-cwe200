import { Injectable, BadRequestException } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import { CodeQlParserService } from '../code-ql/codql-parser-service';

import { ConfigService } from '@nestjs/config';
import { CodeQlService } from 'src/code-ql/code-ql.service';
import { BertService } from 'src/bert/bert.service';
import { LLMService } from 'src/llm/llm.service';
import { JavaParserService } from 'src/parsers/implementations/java-parser.service';
import { AnalyzeRequestDto } from 'src/types/analysis-config.type';
import { LanguageAnalyzerFactory } from 'src/language-analysis/factories/language-analyzer.factory';

import { SensitiveVariables } from 'src/templates/SensitiveVariables';
import { SensitiveComments } from 'src/templates/SensitiveComments';
import { SensitiveStrings } from 'src/templates/SensitiveStrings';
import { Sinks } from 'src/templates/Sinks';

import * as path from 'path';

/**
 * Service responsible for analyzing projects in multiple programming languages,
 * detecting sensitive information, and verifying data flows using various engines and tools.
 * Now supports extensible language analysis through the LanguageAnalyzerFactory.
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
        private codeqlService: CodeQlService,
        private languageAnalyzerFactory: LanguageAnalyzerFactory,
    ) {
        this.projectsPath = this.configService.get<string>('CODEQL_PROJECTS_DIR');
    }

    /**
     * Run analysis on a project using language-specific analyzers
     * This is the main function that kicks off the analysis process.
     * It performs the following steps:
     * 1. Get the appropriate language analyzer from the factory
     * 2. Validate the project for the specified language
     * 3. Discover source files for the language
     * 4. [Attack Surface Detection Engine] Parse the files for variables, strings, comments, and method calls
     * 5. [Attack Surface Detection Engine] Detect sensitive information using BERT
     * 6. [Attack Surface Detection Engine] Read in BERT results from data.json
     * 7. [Attack Surface Detection Engine] Save the sensitive information to .yml files
     * 8. [Exposure Analysis Engine] Create a CodeQL database
     * 9. [Exposure Analysis Engine] Run CWE queries
     * 10. [Exposure Analysis Engine] Save the Dataflow Tree
     * 11. [Flow Verification Engine] Verify Data Flows if any flows are detected
     * 12. [Flow Verification Engine] Update the SARIF file
     * 13. Print the time taken for each step
     * 14. Return the results in the specified format (CSV or SARIF)
     * @param analyzeDto Data transfer object with project name, language, and optional parameters
     */
    async runAnalysis(analyzeDto: AnalyzeRequestDto) {
        const sourcePath = path.join(this.projectsPath, analyzeDto.project);
        this.queryPath = this.configService.get<string>('QUERY_DIR') + analyzeDto.language;

        
        // Get the appropriate language analyzer from the factory
        let languageAnalyzer;
        try {
            languageAnalyzer = this.languageAnalyzerFactory.getAnalyzer(analyzeDto.language);
            if (analyzeDto.javaVersion) {
                languageAnalyzer.setVersion(analyzeDto.javaVersion);
            }
        } catch (error) {
            throw new BadRequestException(error.message);
        }

        // Discover source files using the language-specific analyzer
        let sourceFiles;
        await this.executeStep(`Discovering ${analyzeDto.language} source files`, async () => {
            sourceFiles = await languageAnalyzer.discoverSourceFiles(sourcePath);
            console.log(`Found ${sourceFiles.length} ${analyzeDto.language} files`);
        });

        // Parse files using the language-specific parser
        await this.executeStep('Parsing files for variables, strings, comments, and method calls.', async () => {
            await languageAnalyzer.parseSourceFiles(sourceFiles, sourcePath);
        });

        await this.executeStep('Detecting sensitive info using BERT.', async () => {
            await this.bertService.getBertResponse(sourcePath, "attack_surface_detection.py");
        });

        let data = null;
        await this.executeStep('Reading in BERT results from data.json.', async () => {
            data = this.fileUtilService.parseJSONFile(path.join(sourcePath, 'data.json'));
        });

        await this.executeStep('Saving the sensitive info to .yml files.', async () => {
            await this.saveSensitiveInfo(data);
        });

        // Create language-specific CodeQL database
        await this.executeStep('Creating CodeQL database.', async () => {
            await languageAnalyzer.createCodeQLDatabase(sourcePath, {
                project: analyzeDto.project,
                language: analyzeDto.language,
                extension: analyzeDto.extension
            });
        });

        // Run language-specific CWE queries
        await this.executeStep('Running CWE queries.', async () => {
            await languageAnalyzer.runCWEQueries(sourcePath, {
                project: analyzeDto.project,
                language: analyzeDto.language,
                extension: analyzeDto.extension
            });
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

        if (analyzeDto.extension === 'csv') {
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

    /**
     * Future method for running analysis with language-specific strategies.
     * This demonstrates how the system will work when multiple languages are supported.
     * TODO: Replace runAnalysis with this method once language analyzers are fully implemented.
     */
    async runAnalysisWithLanguageSupport(analyzeDto: AnalyzeRequestDto) {
        // This is how the system will work in the future:
        // 
        // 1. Get language-specific analyzer from factory
        // const analyzer = this.languageAnalyzerFactory.getAnalyzer(analyzeDto.language);
        // 
        // 2. Validate project has files for the specified language
        // const sourcePath = path.join(this.projectsPath, analyzeDto.project);
        // const isValidProject = await analyzer.validateProject(sourcePath);
        // if (!isValidProject) {
        //     throw new BadRequestException(`No ${analyzeDto.language} files found in project ${analyzeDto.project}`);
        // }
        // 
        // 3. Discover language-specific source files
        // const sourceFiles = await analyzer.discoverSourceFiles(sourcePath);
        // 
        // 4. Parse files using language-specific parser
        // await this.executeStep('Parsing files for variables, strings, comments, and method calls.', async () => {
        //     await analyzer.parseSourceFiles(sourceFiles, sourcePath);
        // });
        // 
        // 5. Create language-specific CodeQL database
        // await this.executeStep('Creating CodeQL database.', async () => {
        //     await analyzer.createCodeQLDatabase(sourcePath, {
        //         project: analyzeDto.project,
        //         language: analyzeDto.language,
        //         languageVersion: analyzeDto.javaVersion, // This would be more generic
        //         extension: analyzeDto.extension
        //     });
        // });
        // 
        // 6. Run language-specific CWE queries
        // await this.executeStep('Running CWE queries.', async () => {
        //     await analyzer.runCWEQueries(sourcePath, config);
        // });
        //
        // The rest of the pipeline (BERT analysis, flow verification) would remain the same
        // but could also be made language-aware in the future.
        
        throw new Error('Multi-language support not yet implemented. Use runAnalysis for Java projects.');
    }

    /**
     * Get list of supported programming languages
     * @returns Array of supported language names
     */
    async getSupportedLanguages(): Promise<string[]> {
        return this.languageAnalyzerFactory.getSupportedLanguages();
    }
}

