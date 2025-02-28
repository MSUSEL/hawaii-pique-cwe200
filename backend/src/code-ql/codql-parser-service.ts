import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import * as path from 'path'
import * as os from 'os';
import { parse} from 'csv-parse'
import { promises as fs } from 'fs';


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
        
        // This is just used for testing, it will save the data flow tree to a file, but slows everything down.
        // this.saveDataFlowTree(filePath, project)
    
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

            console.log(flowMaps)            
            return flowMaps; // Return all flow maps (one for each codeFlow)
        } catch (error) {
            console.error('Error processing code flows:', error);
            return [];
        }
    }
    async saveDataFlowTree(project: string) {
        const sarifPath = path.join(project, 'result.sarif');
        const data = await this.fileService.readJsonFile(sarifPath);
        const results = data.runs[0].results;
        const flowMapsByCWE: { [cwe: string]: any[] } = {}; // Group by CWE
      
        try {
          for (let i = 0; i < results.length; i++) {
            const result = results[i];
            const cwe = result.ruleId.split('/').pop(); // Extract CWE from ruleId
            if (!cwe) continue; // Skip if no CWE is found
      
            if (result.codeFlows) {
              // Ensure there's an array for this CWE
              if (!flowMapsByCWE[cwe]) {
                flowMapsByCWE[cwe] = [];
              }
              const codeFlowsList = result.codeFlows;
              for (let j = 0; j < codeFlowsList.length; j++) {
                const codeFlows = codeFlowsList[j].threadFlows[0].locations;
      
                // Retrieve the flow map for each code flow
                const flowMap = await this.buildDataFlowMap(codeFlows, project);
      
                // Format the flow map to be more human-readable
                const humanReadableFlowMap = Object.entries(flowMap).map(([index, node]) => ({
                  step: Number(index),  // Step number for readability
                  variableName: node.message,
                  uri: node.uri,
                  type: node.type,
                  code: node.code
                }));
      
                // Get the fileName from the last element in the flow (without modifying the array)
                const fileName = humanReadableFlowMap[humanReadableFlowMap.length - 1].uri.split('/').pop();
      
                // Check if there is already an entry for this resultIndex
                let resultEntry = flowMapsByCWE[cwe].find(entry => entry.resultIndex === i + 1);
                if (!resultEntry) {
                  resultEntry = {
                    resultIndex: i,
                    fileName: fileName,
                    flows: []
                  };
                  flowMapsByCWE[cwe].push(resultEntry);
                }
      
                // Add this flow to the result's flows array
                resultEntry.flows.push({
                  codeFlowIndex: j,
                  flow: humanReadableFlowMap,
                });
              }
            }
          }
      
          // Save the grouped flow maps in human-readable format
          const outputFilePath = path.join(project, 'flowMapsByCWE.json');
          await this.fileService.writeToFile(outputFilePath, JSON.stringify(flowMapsByCWE, null, 2));
          // console.log(`Data flow map grouped by CWE saved to ${outputFilePath}`);
      
        } catch (error) {
          // console.error('Error processing code flows:', error);
        }
      }
      

    async buildDataFlowMap(codeFlows: any[], project: string): Promise<{ [key: number]: FlowNode }> {
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
                    const lines = data.split('\n');

                    // Calculate the range of lines we need (3 lines above and 3 below)
                    const startExtractLine = Math.max(0, startLine - 4); // 3 lines above (0-based index)
                    const endExtractLine = Math.min(lines.length, startLine + 2); // 3 lines below (inclusive)

                    // Extract the required lines
                    const surroundingLines = lines.slice(startExtractLine, endExtractLine + 1);

                    // Ensure startColumn and endColumn are within valid bounds on the specified line
                    const targetLine = surroundingLines[3]; // The actual line (3rd in extracted array)
                    const validStartColumn = Math.max(0, startColumn - 1);
                    const validEndColumn = Math.min(endColumn - 1, targetLine.length);

                    // Update message if valid columns are provided
                    if (validStartColumn <= validEndColumn) {
                        message = targetLine.slice(validStartColumn, validEndColumn);
                    }

                    // Join surrounding lines for full context
                    const codeContext = surroundingLines.join('\n');

                    // Store this in the flow map
                    flowMap[flowIndex] = {
                        message: message,  // Use the updated message (either from SARIF or extracted from the file)
                        uri: uri,  // Keep the original URI for reference
                        startLine: startLine,
                        startColumn: startColumn,
                        endColumn: endColumn,
                        endLine: endLine,
                        type: type,
                        code: codeContext  // Add surrounding code as the context
                    };

                } catch (error) {
                    console.error(`Error reading file ${fullPath}:`, error);
                    // Optionally handle the error here
                }
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
    async getcsvResults(sourcePath: string) {
        // Construct the SARIF file path
        const sarifPath = path.join(sourcePath, 'result.sarif');
        const csvPath = path.join(sourcePath, 'result.csv'); // Output CSV file
    
        try {
            // Read the SARIF file content
            const fileData = await this.fileService.readJsonFile(sarifPath);
    
            // Initialize an array for CSV rows
            const csvData: any[] = [];
    
            // Process SARIF results
            const results = fileData.runs[0]?.results || [];
            for (const result of results) {
                const message = result.message.text.split('\n')[0]; // First line of the message
                const location = result.locations?.[0]?.physicalLocation;
                const ruleId = result.ruleId.split('/').pop() || 'N/A';
    
                const path = location?.artifactLocation?.uri || 'N/A';
                const region = location?.region || {};
                const startLine = region.startLine || 'N/A';
                const endLine = region.endLine || startLine;
                const startColumn = region.startColumn || 'N/A';
                const endColumn = region.endColumn || 'N/A';
    
                // Push extracted data as a row
                const dataString = `${ruleId},${path},${startLine},${startColumn}`;
                csvData.push(dataString);
            }

            await fs.writeFile(csvPath, csvData.join('\n'), 'utf-8');

            const csv = path.join(sourcePath, 'result.csv');
            const data = await this.fileService.readFileAsync(csv);
            return {data};
    
        } catch (error) {
            console.error('Error processing SARIF file:', error.message);
            throw error;
        }
    }
    
    
    parseResults(rules: any[], results: any[],sourcePath:string) {
        var resultList: Array<{ name: string; fullPath: string, files:any[] }> = [];
        for (let i = 0; i < results.length; i++) {
            let result = results[i];
            let CWE = result.message.text.split(':')[0];
            var filePath=result.locations[0]?.physicalLocation.artifactLocation.uri;
            var fullPath = this.correctPath(this.fileService.processFilePath(sourcePath,filePath));

            var fileIndex = resultList.findIndex(
                (file) => this.correctPath(file.fullPath) == fullPath,
            );
            var file: any = null;
            if (fileIndex == -1) {
                file = {
                    cwe: CWE,
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
                cwe: CWE,
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
    type: string,
    code: string
}

export interface Region {
    startLine: number;
    startColumn: number;
    endColumn: number;
    endLine: number;
  }