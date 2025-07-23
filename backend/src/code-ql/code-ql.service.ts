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
    runChildProcess(args: string[]): Promise<void> {
    return new Promise((resolve, reject) => {
        const childProcess = spawn('codeql', args);

        let stdout = '';
        let stderr = '';

        childProcess.stdout.on('data', (data) => {
            const text = data.toString();
            stdout += text;
            console.log(text);
            this.eventsGateway.emitDataToClients('data', text);
        });

        childProcess.stderr.on('data', (data) => {
            const text = data.toString();
            stderr += text;
            console.error(text);
            this.eventsGateway.emitDataToClients('data', text);
        });

        childProcess.on('exit', (code, signal) => {
            const result = `process CodeQL exited with code ${code} and signal ${signal}`;
            console.log(result);
            this.eventsGateway.emitDataToClients('data', result);

            if (code !== 0) {
                return reject(new Error(`CodeQL failed with code ${code}, stderr: ${stderr}`));
            }
            resolve();
        });

        childProcess.on('error', (error) => {
            console.error('Failed to start CodeQL process:', error);
            reject(error);
        });
    });
}


    /**
     * Create a CodeQL database for the given project.
     * @param sourcePath The path to the source code directory.
     * @param createCodeQlDto The DTO containing project information.
     */

    async createDatabase(sourcePath: string, createCodeQlDto: any): Promise<void> {
    const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
    await this.fileUtilService.removeDir(db);

    const createDbCommand = [
    'database', 'create', db,
    `--language=java`,
    `--source-root=${sourcePath}`
    ];

    await this.runChildProcess(createDbCommand);
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

        const args = [
            'database', 'analyze',
            db,
            'custom-codeql-queries',
            `--format=${format}`,
            '--rerun',
            `--output=${outputPath}`,
            `--threads=${threads}`,
            `--ram=${ramAllocationMB}`,
            `--search-path=${path.resolve(queryPath, '..')}`
        ];

        console.log(args);
        await this.runChildProcess(args);
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
    // Decode the whole vulnerabilityId upfront
        const sourcePath = path.join(this.projectsPath, project);
        return await this.parserService.getDataFlowTree(sourcePath, index);
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
}