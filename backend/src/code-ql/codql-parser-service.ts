import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import * as path from 'path'
import * as os from 'os';


@Injectable()
export class CodeQlParserService {
    constructor(private fileService: FileUtilService) {}

    async getSarifResults(sourcePath:string) {
        var sarifPath = path.join(sourcePath,'result.sarif');
        var data = await this.fileService.readJsonFile(sarifPath);
        var rules = data.runs[0].tool.driver.rules;
        var results = data.runs[0].results;
        var rulesTree = this.parseRules(rules, results,sourcePath);
        var locationsTree=this.parseResults(rules,results,sourcePath)
        return {rulesTree,locationsTree};
    }

    async getDataFlowTree(filePath: string, project: string, index: string) {
        const sarifPath = path.join(project, 'result.sarif');
        const data = await this.fileService.readJsonFile(sarifPath);
        const result = data.runs[0].results[index];
    
        try {
            // Try to access all codeFlows
            const codeFlowsList = result.codeFlows || [];
            const flowMaps = [];
    
            // Loop through each codeFlow and build the flow map for each
            for (let i = 0; i < codeFlowsList.length; i++) {
                const codeFlows = codeFlowsList[i].threadFlows[0].locations;
                const flowMap = await this.buildDataFlowMap(codeFlows, project);
                flowMaps.push(flowMap);
            }
    
            return flowMaps; // Return all flow maps (one for each codeFlow)
        } catch (error) {
            console.error('Error processing code flows:', error);
            return [];
        }
    }
    
    async buildDataFlowMap(codeFlows: any[], project): Promise<{ [key: number]: FlowNode }> {
        const flowMap: { [key: number]: FlowNode } = {};
    
        // Use a loop with `await` to ensure file loading is awaited for each code flow
        for (let flowIndex = 0; flowIndex < codeFlows.length; flowIndex++) {
            const codeFlow = codeFlows[flowIndex];
            console.log(`Processing Code Flow #${flowIndex + 1}`);
    
            const location = codeFlow.location;
            if (location) {
                const physicalLocation = location.physicalLocation;
                const uri = physicalLocation.artifactLocation.uri;
                const startLine = physicalLocation.region.startLine;
                const startColumn = physicalLocation.region.startColumn;
                const endColumn = physicalLocation.region.endColumn;
                const endLine = physicalLocation.region.endLine || startLine;
    
                // Normalize the file path for the current OS
                const normalizedUri = path.normalize(uri);
                const fullPath = path.join(project, normalizedUri);
    
                let message = location.message.text;  // Default message from SARIF file
                const type = message.length > 1 ? message.split(':').slice(1).join(':').trim() : '';
                console.log(`Message: ${message}`);

                // Await the file read to ensure we get the file data before continuing
                try {
                    const data = await this.fileService.readFileAsync(fullPath);
    
                    // Extract the specific line from the file
                    const line = data.split('\n')[startLine - 1];
    
                    // Ensure startColumn and endColumn are within valid bounds
                    const validStartColumn = Math.max(0, startColumn - 1);
                    const validEndColumn = Math.min(endColumn - 1, line.length );
    
                    // Extract the substring between validStartColumn and validEndColumn
                    if (validStartColumn <= validEndColumn) {
                        message = line.slice(validStartColumn, validEndColumn);
                    } else {
                    }
    
                } catch (error) {
                    console.error(`Error reading file ${fullPath}:`, error);
                    // Optionally handle the error here
                }
    
                // Create a key-value pair for this location
                flowMap[flowIndex] = {
                    message: message,  // Use the updated message (either from SARIF or extracted from the file)
                    uri: uri,  // Keep the original URI for reference
                    startLine: startLine,
                    startColumn: startColumn,
                    endColumn: endColumn,
                    endLine: endLine,
                    type: type,
                };
            }
        }
    
        return flowMap; // Return the flow map after all files have been read
    }
    
    

    
    

    parseRules(rules: any[], results: any[], sourcePath: string) {
        const rulesMap = new Map();
        const fileMap = new Map();  // Map to track files and their associated rules
        let overallIndex = 0;  // To track the index across all results
    
        for (let i = 0; i < rules.length; i++) {
            let rule = rules[i];
            let ruleKey = `CWE-${rule.id.split("/").pop()}`;
    
            // Filter results based on ruleIndex
            var files = results
                .filter((item) => item.ruleIndex == i)
                .map((file) => {
                    const mappedFile = {
                        name: this.fileService.getFilenameFromPath(
                            file.locations[0]?.physicalLocation.artifactLocation.uri
                        ),
                        fullPath: this.correctPath(this.fileService.processFilePath(
                            sourcePath,
                            file.locations[0]?.physicalLocation.artifactLocation.uri
                        )),
                        message: file.message.text,
                        region: file.locations[0]?.physicalLocation.region,
                        location: file.locations[0]?.physicalLocation.region.startLine.toString(),
                        index: overallIndex.toString(),  // Set the index over all results
                    };
    
                    overallIndex++;  // Increment after each result is processed
                    return mappedFile;
                })
                .filter(file => {
                    // Construct a unique identifier for the file based on its location and line
                    const fileIdentifier = `${file.fullPath}:${file.region.startLine}`;
    
                    // Check if this file has already been processed under another rule
                    if (fileMap.has(fileIdentifier)) {
                        return false;  // Skip this file as it's already associated with a rule
                    } else {
                        fileMap.set(fileIdentifier, ruleKey);  // Mark this file as processed under this rule
                        return true;
                    }
                });
    
            // Check if the ruleKey already exists in the map
            if (rulesMap.has(ruleKey)) {
                const existingEntry = rulesMap.get(ruleKey);
                existingEntry.files = existingEntry.files.concat(files);
                rulesMap.set(ruleKey, existingEntry);
            } else {
                // Create a new entry if it does not exist
                var object = {
                    name: ruleKey,
                    type: rule.defaultConfiguration.level,
                    message: rule.fullDescription ? rule.fullDescription.text : rule.shortDescription.text,
                    files: files,
                };
                if (object.files.length) rulesMap.set(ruleKey, object);
            }
        }
    
        // Convert the Map values to an array since the final result expects an array
        return Array.from(rulesMap.values());
    }
    
    
    
    
    

    parseResults(rules: any[], results: any[],sourcePath:string) {
        var resultList: Array<{ name: string; fullPath: string, files:any[] }> = [];
        for (let i = 0; i < results.length; i++) {
            let result = results[i];
            var filePath=result.locations[0]?.physicalLocation.artifactLocation.uri;
            var fullPath = this.correctPath(this.fileService.processFilePath(sourcePath,filePath));

            var fileIndex = resultList.findIndex(
                (file) => this.correctPath(file.fullPath) == fullPath,
            );
            var file: any = null;
            if (fileIndex == -1) {
                file = {
                    name: this.fileService.getFilenameFromPath(filePath),
                    fullPath: this.correctPath(this.fileService.processFilePath(sourcePath,filePath)),
                    files:[]
                };
                resultList.push(file)
            }else{
                file=resultList[fileIndex]
            }
            var region=result.locations[0]?.physicalLocation.region;
            region.endLine=region.endLine?region.endLine:region.startLine;
            var rule = rules[result.ruleIndex];
            file.files.push({
                name: rule.shortDescription.text,
                type: rule.defaultConfiguration.level,
                message: result.message.text,
                region: region,
            })
        }
        return resultList;
    }

    correctPath(filePath: string) {
        if (os.platform() === 'win32') {
            return filePath.replace(/\//g, '\\');
        }
        // Normalize path for Linux and macOS
        else {
            return filePath.replace(/\\/g, '/');
        }
    }
}


const cweDescriptions = new Map([
    ["CWE-200", "Exposure of Sensitive Information to an Unauthorized Actor"],
    ["CWE-201", "Insertion of Sensitive Information Into Sent Data"],
    ["CWE-204", "Observable Response Discrepancy"],
    ["CWE-205", "Observable Behavioral Discrepancy"],
    ["CWE-208", "Observable Timing Discrepancy"],
    ["CWE-209", "Generation of Error Message Containing Sensitive Information"],
    ["CWE-210", "Self-generated Error Message Containing Sensitive Information"],
    ["CWE-211", "Externally-Generated Error Message Containing Sensitive Information"],
    ["CWE-213", "Exposure of Sensitive Information Due to Incompatible Policies"],
    ["CWE-214", "Invocation of Process Using Visible Sensitive Information"],
    ["CWE-215", "Insertion of Sensitive Information Into Debugging Code"],
    ["CWE-497", "Exposure of Sensitive System Information to an Unauthorized Control Sphere"],
    ["CWE-531", "Inclusion of Sensitive Information in Test Code"],
    ["CWE-532", "Insertion of Sensitive Information into Log File"],
    ["CWE-535", "Exposure of Information Through Shell Error Message"],
    ["CWE-536", "Servlet Runtime Error Message Containing Sensitive Information"],
    ["CWE-537", "Java Runtime Error Message Containing Sensitive Information"],
    ["CWE-538", "Insertion of Sensitive Information into Externally-Accessible File or Directory"],
    ["CWE-540", "Inclusion of Sensitive Information in Source Code"],
    ["CWE-541", "Inclusion of Sensitive Information in an Include File"],
    ["CWE-548", "Exposure of Information Through Directory Listing"],
    ["CWE-550", "Server-generated Error Message Containing Sensitive Information"],
    ["CWE-598", "Use of GET Request Method With Sensitive Query Strings"],
    ["CWE-615", "Inclusion of Sensitive Information in Source Code Comments"],
    ["CWE-651", "Exposure of WSDL File Containing Sensitive Information"],
]);

export interface FlowNode {
    message: string
    uri: string,
    startLine: string,
    startColumn: string,
    endColumn: string,
    endLine: string,
    type: string
}

export interface Region {
    startLine: number;
    startColumn: number;
    endColumn: number;
    endLine: number;
  }