import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import { exec, spawn } from 'child_process';
import { CodeQlParserService } from './codql-parser-service';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { ConfigService } from '@nestjs/config';
import { EventsGateway } from 'src/events/events.gateway';


/**
 * CodeQlService is responsible for running CodeQL commands and analyzing Java projects.
 * It provides methods to create databases, perform backward slicing, run CWE queries,
 */
@Injectable()
export class CodeQlService {
    projectsPath: string;
    queryPath: string;
    constructor(
        private configService: ConfigService,
        private parserService: CodeQlParserService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
    ) {
        this.projectsPath = this.configService.get<string>('CODEQL_PROJECTS_DIR');
        this.queryPath = this.configService.get<string>('QUERY_DIR');
      }


    /**
     * Execute command string using codeql
     *
     * @param codeQlCommand formatted codeql arguments
     */
    runChildProcess(codeQlCommand: string): Promise<void> {
        const commands = codeQlCommand.split(' ');
        return new Promise((resolve, reject) => {
            // create new codeql process
            let childProcess = spawn('codeql', commands);

            // report stdout
            childProcess.stdout.on('data', (data) => {
                console.log(data.toString());
                this.eventsGateway.emitDataToClients('data', data.toString())
            });

            // report stderr
            childProcess.stderr.on('data', (data) => {
                console.log(data.toString());
                this.eventsGateway.emitDataToClients('data', data.toString())
            });

            // report results after finishing
            const self = this;
            childProcess.on('exit', function (code, signal) {
                const result = "process CodeQl exited with code " + code + " and signal " + signal;
                // if (code !== 0) {
                //     reject(result);
                //     throw new Error(result);
                // }
                console.log(result);
                self.eventsGateway.emitDataToClients('data', result.toString())
                resolve();
            });

            // report errors
            childProcess.on('error', (error) => {
                console.log(error);
                reject(error);
            });
        });
    }

    /**
     * Create a CodeQL database for the given project.
     * @param sourcePath The path to the source code directory.
     * @param createCodeQlDto The DTO containing project information.
     */
    async createDatabase(sourcePath: string, createCodeQlDto: any) {
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
        await this.fileUtilService.removeDir(db);
        const createDbCommand = `database create ${db} --language=java --source-root=${sourcePath}`;
        console.log(createDbCommand);
        await this.runChildProcess(createDbCommand);
    }

    /**
     * [Unused] Perform backward slicing on the given project.
     * @param sourcePath The path to the source code directory.
     * @param createCodeQlDto The DTO containing project information.
     */
    async performBackwardSlicing(sourcePath: string, createCodeQlDto: any) {
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
        const extension = 'sarif';
        const format = 'sarifv2.1.0';
        const outputFileName = 'backwardslice';
        const outputPath = path.join(sourcePath, `${outputFileName}.${extension}`);
        const threads = 12;
        const totalMemoryMB = os.totalmem() / (1024 * 1024);  // Total system memory in MB
        const ramAllocationMB = Math.floor(totalMemoryMB * 0.8);  // 80% of total memory
        const queryPath = path.join(this.queryPath, 'ProgramSlicing', 'Variables');

        // Command to run backward slicing
        const analyzeDbCommand =
            `database analyze ${db} ` +
            `--format=${format} --rerun ` +
            `--output=${outputPath} ${queryPath} ` +
            `--max-paths=1 --sarif-add-snippets=true ` +
            `--threads=${threads} --ram=${ramAllocationMB}`;
        await this.runChildProcess(analyzeDbCommand);

    }

    /**
     * Run CWE queries located in "codeql/codeql-custom-queries-java" on the given project.
     * @param sourcePath The path to the source code directory.
     * @param createCodeQlDto The DTO containing project information.
     */
    async runCWEQueries(sourcePath: string, createCodeQlDto: any) {
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');
        const extension = 'sarif';
        const format = 'sarifv2.1.0';
        const outputFileName = 'result';
        const outputPath = path.join(sourcePath, `${outputFileName}.${extension}`);
        const threads = 12;
        const totalMemoryMB = os.totalmem() / (1024 * 1024);
        const ramAllocationMB = Math.floor(totalMemoryMB * 0.8);
        const queryPath = this.queryPath; // e.g., codeql/codeql-custom-queries-java
    
        const analyzeDbCommand =
            `database analyze ${db} custom-codeql-queries ` +
            `--format=${format} ` +
            `--rerun ` +
            `--output=${outputPath} ` +
            `--threads=${threads} ` +
            `--ram=${ramAllocationMB} ` +
            `--search-path=${path.resolve(queryPath, '..')}`;
    
        console.log(analyzeDbCommand);
        await this.runChildProcess(analyzeDbCommand);
    }
    

    /**
     * Get SARIF results for a specific project that has already been analyzed.
     * @param project The name of the project to get SARIF results for.
     * @returns The SARIF results or an error message if the file does not exist.
     */
    async getSarifResults(project: string) {
        const sourcePath = path.join(this.projectsPath, project);
        console.log(sourcePath);
        if (!fs.existsSync(path.join(sourcePath, 'result.sarif'))) {
            return { error: 'Results file does not exist' };
        }
        let data = await this.parserService.getSarifResults(sourcePath);
        return await this.parserService.getSarifResults(sourcePath);
    }

    /**
     * Get the data flow tree for a specific vulnerability in a project.
     * @param vulnerabilityId The ID of the vulnerability to get the data flow tree for.
     * @param project The name of the project to get the data flow tree for.
     * @param index The index of the flow node to get.
     * @returns The data flow tree for the specified vulnerability.
     */
    async getDataFlowTree(vulnerabilityId: string, project: string, index: string) {
        {

            // Find the first occurrence of the project in the vulnerabilityId
            let projectIndex = vulnerabilityId.indexOf(project);

            // If the project is found, slice after the first occurrence of the project
            if (projectIndex !== -1) {
                // Slice from the first occurrence of the project
                vulnerabilityId = vulnerabilityId.slice(projectIndex);

                // If there is a second occurrence of the project, remove everything before it
                let secondProjectIndex = vulnerabilityId.indexOf(project, project.length);
                if (secondProjectIndex !== -1) {
                    vulnerabilityId = vulnerabilityId.slice(0, secondProjectIndex - 1) + vulnerabilityId.slice(secondProjectIndex + project.length);
                }
            }

            // console.log(vulnerabilityId, project, index);

            const sourcePath = path.join(this.projectsPath, project);
            return await this.parserService.getDataFlowTree(vulnerabilityId, sourcePath, index);
        }
    }

    /**
     * This function applies the labels from the labeling data to the codeFlows 
     * in the flowMapsByCWE.json file. These labels are used to train the BERT model.
     * Used in the verification step of the detections produced by CodeQL.
     * @param labelData Data from the front-end labeling tool.
     */
    async labelFlows(labelData: any) {
        const sourcePath = path.join(this.projectsPath, labelData.project);
        // await this.countFlowsBetweenJsonAndSarif(sourcePath)
        var codeFlowsPath = path.join(sourcePath, 'flowMapsByCWE.json');
        var codeFlows = await this.fileUtilService.readJsonFile(codeFlowsPath);
        // Get the vulnerabilityId from the first result
        for (let i = 0; i < labelData.vulnerabilities.length; i++) {
            let vulIndex = labelData.vulnerabilities[i].vulnerabilityId

            // Get the flows from the vulnerabilityId
            for (let j = 0; j < labelData.vulnerabilities[i].flows.length; j++) {
                let flowIndex = labelData.vulnerabilities[i].flows[j].flowIndex
                let label = labelData.vulnerabilities[i].flows[j].label

                if (label === "No" || label === "Yes") {

                    Object.keys(codeFlows).forEach(cwe => {
                        // Check if the vulIndex from the labeling matches the resultIndex from the codeFlows
                        // Because not all results in the original SARIF file have codeFlows
                        // The index of the result in the codeFlows isn't always the same as the vulIndex from the labeling.
                        // So we have to inefficiently loop through all the codeFlows to find the correct one.
                        for (let res of codeFlows[cwe]) {
                            if (res.resultIndex === Number(vulIndex)) {
                                // Check if the flowIndex from the labeling matches the codeFlowIndex from the codeFlows
                                for (let flow of res.flows) {
                                    if (flow.codeFlowIndex === Number(flowIndex)) {
                                        flow.label = label
                                        console.log("Added label to CWE " + cwe + " vulIndex " + vulIndex + " flowIndex " + flowIndex + " label " + label)
                                    }
                                }
                            }
                        }
                    });
                } else {
                    console.log("-- No label for vulIndex " + vulIndex + " flowIndex " + flowIndex + " label " + label)
                }

            }
        }
        console.log("This should be last")
        // Save the updated codeFlows back to the file
        await this.fileUtilService.writeToFile(path.join("..", "testing", "Labeling", "Flow Verification", "FlowData", String(labelData.project + ".json")), JSON.stringify(codeFlows, null, 2));
    }


    /**
     * Adds labels to the flows in the SARIF file.
     *
     * This function processes the SARIF file after BERT has determined which flows
     * are valid based on the flowmap data, it adds corresponding labels to those flows.
     * So that flows labeled as No (Not sensitive) are not included in the final report.
     *
     * @param sourcePath The path to the project directory.
     */
    async updateSarif(sourcePath: string) {
        const codeFlowsPath = path.join(sourcePath, 'flowMapsByCWE.json');
        const codeFlows = await this.fileUtilService.readJsonFile(codeFlowsPath);

        const sarifPath = path.join(sourcePath, 'result.sarif');
        const sarifdata = await this.fileUtilService.readJsonFile(sarifPath);

        Object.keys(codeFlows).forEach(cwe => {
            const results = codeFlows[cwe];
            // Iterate over the results for each CWE
            for (let j = 0; j < results.length; j++) {
                const resultIndex = results[j].resultIndex;
                const flows = results[j].flows;
                // Iterate over the flows for each result
                for (let i = 0; i < flows.length; i++) {
                    const flowIndex = flows[i].codeFlowIndex;
                    const label = flows[i].label;
                    const sarifResult = sarifdata.runs[0].results[resultIndex];

                    // If there is a codeFlow, update it
                    if (sarifResult.codeFlows && sarifResult.codeFlows[flowIndex]) {
                        sarifResult.codeFlows[flowIndex].label = label;
                    }
                    // Otherwise, if the result was generated from locations, update the location directly
                    else if (sarifResult.locations) {
                        // Here, you can choose how to store the label. For example, add a new property to each location:
                        sarifResult.locations.forEach(location => {
                            location.label = label;
                        });
                    }
                }
            }
        });
        // Save the updated sarifdata back to the file
        await this.fileUtilService.writeToFile(sarifPath, JSON.stringify(sarifdata, null, 2));
    }

    /**
     * Create a language-agnostic CodeQL database
     * This method demonstrates how to make database creation language-aware
     * @param sourcePath - Path to source directory
     * @param language - Programming language
     * @param projectName - Name of the project
     * @param config - Additional configuration
     */
    async createLanguageDatabase(sourcePath: string, language: string, projectName: string, config: any = {}) {
        const db = path.join(sourcePath, projectName + 'db');
        await this.fileUtilService.removeDir(db);
        
        // Map programming languages to CodeQL language identifiers
        const codeqlLanguageMap: Record<string, string> = {
            'java': 'java',
            'python': 'python',
            'javascript': 'javascript',
            'typescript': 'javascript', // TypeScript is analyzed as JavaScript in CodeQL
            'csharp': 'csharp',
            'cpp': 'cpp',
            'c': 'cpp', // C is analyzed as C++ in CodeQL
            'go': 'go',
            'ruby': 'ruby',
            'swift': 'swift',
        };

        const codeqlLanguage = codeqlLanguageMap[language.toLowerCase()];
        if (!codeqlLanguage) {
            throw new Error(`CodeQL does not support language: ${language}`);
        }

        const createDbCommand = `database create ${db} --language=${codeqlLanguage} --source-root=${sourcePath}`;
        console.log(`Creating CodeQL database for ${language}:`, createDbCommand);
        await this.runChildProcess(createDbCommand);
    }

    /**
     * Get supported languages for CodeQL analysis
     */
    getSupportedCodeQLLanguages(): string[] {
        return ['java', 'python', 'javascript', 'typescript', 'csharp', 'cpp', 'c', 'go', 'ruby', 'swift'];
    }
}