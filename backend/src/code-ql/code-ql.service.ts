import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import { exec, spawn } from 'child_process';
import { CodeQlParserService } from './codql-parser-service';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

import { ConfigService } from '@nestjs/config';


import { EventsGateway } from 'src/events/events.gateway';
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


    async createDatabase(sourcePath: string, createCodeQlDto: any) {
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
        await this.fileUtilService.removeDir(db);
        const createDbCommand = `database create ${db} --language=java --source-root=${sourcePath}`;
        console.log(createDbCommand);
        await this.runChildProcess(createDbCommand);
    }

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

    async runCWEQueries(sourcePath: string, createCodeQlDto: any) {
        const db = path.join(sourcePath, createCodeQlDto.project + 'db');   // path to codeql database
        const extension = 'sarif';
        const format = 'sarifv2.1.0';
        const outputFileName = 'result';
        const outputPath = path.join(sourcePath, `${outputFileName}.${extension}`);
        const threads = 12;
        const totalMemoryMB = os.totalmem() / (1024 * 1024);  // Total system memory in MB
        const ramAllocationMB = Math.floor(totalMemoryMB * 0.8);  // 80% of total memory
        const queryPath = this.queryPath;

        // This is for running all of the queries
        const queryDir = path.resolve(queryPath);
        const excludeDir = path.resolve(path.join(queryPath, 'ProgramSlicing'));

        function collectQlFiles(dir) {
            let results = [];
            const list = fs.readdirSync(dir);
            list.forEach(file => {
                const filePath = path.join(dir, file);
                const stat = fs.statSync(filePath);
                if (stat && stat.isDirectory()) {
                    // Recursively collect .ql files from subdirectories
                    results = results.concat(collectQlFiles(filePath));
                } else if (filePath.endsWith('.ql') && !filePath.startsWith(excludeDir)) {
                    // Include only .ql files and exclude those in the ProgramSlicing directory
                    results.push(filePath);
                }
            });
            return results;
        }

        // Collect all .ql files from the queryDir, excluding ProgramSlicing subdir
        let queriesToRun = collectQlFiles(queryDir);

        // Join the selected queries into a single string
        const queryList = queriesToRun.join(' ');

        // Build the command with the filtered list of queries
        const analyzeDbCommand =
            `database analyze ${db} ` +
            `--format=${format} ` +
            `--rerun ` +
            `--output=${outputPath} ${queryList} ` +
            `--threads=${threads} ` +
            `--ram=${ramAllocationMB}`;
        await this.runChildProcess(analyzeDbCommand);
    }

    async getSarifResults(project: string) {
        const sourcePath = path.join(this.projectsPath, project);
        console.log(sourcePath);
        if (!fs.existsSync(path.join(sourcePath, 'result.sarif'))) {
            return { error: 'Results file does not exist' };
        }
        let data = await this.parserService.getSarifResults(sourcePath);
        return await this.parserService.getSarifResults(sourcePath);
    }

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
        await this.fileUtilService.writeToFile(path.join("..", "testing", "labeling", "FlowData", String(labelData.project + ".json")), JSON.stringify(codeFlows, null, 2));
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
        var codeFlowsPath = path.join(sourcePath, 'flowMapsByCWE.json');
        var codeFlows = await this.fileUtilService.readJsonFile(codeFlowsPath);

        var sarifPath = path.join(sourcePath, 'result.sarif');
        var sarifdata = await this.fileUtilService.readJsonFile(sarifPath);

        Object.keys(codeFlows).forEach(cwe => {

            let results = codeFlows[cwe]
            // Iterate over the results for each CWE
            for (let j = 0; j < results.length; j++) {
                let resultIndex = results[j].resultIndex
                let flows = results[j].flows

                // Iterate over the flows for each result
                for (let i = 0; i < flows.length; i++) {
                    let flowIndex = flows[i].codeFlowIndex
                    let label = flows[i].label

                    // Make sure the resultIndex is the index of the sarif result. 
                    // Not all sarif results have a codeFlow
                    if (sarifdata.runs[0].results[resultIndex].codeFlows[flowIndex]) {
                        sarifdata.runs[0].results[resultIndex].codeFlows[flowIndex].label = label
                    }
                    else {

                    }
                }
            }
        });
        // Save the updated sarifdata back to the file
        await this.fileUtilService.writeToFile(sarifPath, JSON.stringify(sarifdata, null, 2));
    }

}