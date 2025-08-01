import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import * as path from 'path'
import * as os from 'os';
import { parse} from 'csv-parse'
import { promises as fs } from 'fs';
import { start } from 'repl';


@Injectable()
export class CodeQlParserService {
    results: any[] = [];
    constructor(private fileService: FileUtilService) {

    }
    /**
     * This function is used to parse the SARIF file and extract the results.
     * @param sourcePath - The path to the source directory.
     * @returns 
     */
    async getSarifResults(sourcePath: string) {
        const sarifPath = path.join(sourcePath, 'result.sarif');
        const data = await this.fileService.readJsonFile(sarifPath);
        const rules = data.runs[0].tool.driver.rules;
        let results = data.runs[0].results;

        results = this.filterResults(results)
        this.results = results
      

      
        const rulesTree = this.parseRules(rules, results, sourcePath);
        const locationsTree = this.parseResults(rules, results, sourcePath);
        return { rulesTree, locationsTree };
      }
    
    /**
     * This function is used to filter flows out that are likely false positives. 
     * The labeling along with the flows can be found in the flowMapsByCWE.json for each project.
     * @  param results - The results array from the SARIF file.
     * @returns - The filtered results array with only relevant flows.
     */
    filterResults(results) {
    let totalFlowCount = 0;
    let totalFlowRemovedCount = 0;

    results = results.filter(result => {
        let originalCount = 0;
        let filteredFlows = [];

        if (result.codeFlows && result.codeFlows.length > 0) {
            // Handle multi-flow results
            originalCount = result.codeFlows.length;

            filteredFlows = result.codeFlows.filter(flow => {
                return flow.label && flow.label.toLowerCase().trim() === 'yes';
            });

            result.codeFlows = filteredFlows;
        } else if (result.locations && result.locations.length > 0) {
            // Handle single-flow result via locations
            originalCount = result.locations.length;

            filteredFlows = result.locations.filter(location => {
                return location.label && location.label.toLowerCase().trim() === 'yes';
            });

            result.locations = filteredFlows;
        } else {
            // No codeFlows or locations = no usable flow
            return false;
        }

        totalFlowCount += originalCount;
        totalFlowRemovedCount += (originalCount - filteredFlows.length);

        // Keep the result only if at least one flow remains
        return filteredFlows.length > 0;
    });

    console.log(`Filtered out ${totalFlowRemovedCount} flows out of ${totalFlowCount} total flows.`);
    return results;
}

      
    /**
     * This function is used to get all the data flow trees for a specific result index
     * @param filePath - The path to the source directory.
     * @param project - The name of the project to get flow nodes for
     * @param index - The index of the flow node to get
     * @returns - The flow nodes for the specified vulnerability
     */
async getDataFlowTree(project: string, index: string) {
    const result = this.results[index];
    try {
        const flowMaps = [];
        if (result.codeFlows && result.codeFlows.length > 0) {
            // Process each codeFlow
            for (let i = 0; i < result.codeFlows.length; i++) {
                const codeFlows = result.codeFlows[i].threadFlows[0].locations;
                const flowMap = await this.buildDataFlowMap(codeFlows, project);
                flowMaps.push(flowMap);
            }
        } else if (result.locations) {
            // Use result.locations in absence of codeFlows
            const flowMap = await this.buildDataFlowMap(result.locations, project);
            flowMaps.push(flowMap);
        }
        // console.log(flowMaps);
        return flowMaps; // Return all flow maps
    } catch (error) {
        console.error('Error processing data flows:', error);
        return [];
    }
}
    
    /**
     * This function is used to save the data flow trees for each CWE in a separate file.
     * This format is used as input for the Flow Verification Engine.
     * It is more readable and easier to work with than the SARIF file.
     * @param project - The name of the project to save the flow maps for
     */
    async saveDataFlowTree(project: string) {
        const sarifPath = path.join(project, 'result.sarif');
        const data = await this.fileService.readJsonFile(sarifPath);
        const results = data.runs[0].results;
        const flowMapsByCWE: { [cwe: string]: any[] } = {}; // Group by CWE
      
        try {
          for (let i = 0; i < results.length; i++) {
            const result = results[i];
            const cwe = result.ruleId.split('/').pop(); // Extract CWE from ruleId
            if (!cwe){
                console.log('No CWE found for result:', result);
                continue; // Skip if no CWE is found
            } 
      
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
                  step: Number(index), // 0-based step number
                  variableName: node.message,
                  startLine: node.startLine,
                  startColumn: node.startColumn,
                  endLine: node.endLine,
                  endColumn: node.endColumn,
                  uri: node.uri,
                  type: node.type,
                  code: node.code
                }));
      
                // Get the fileName from the last element in the flow (without modifying the array)
                const fileName = humanReadableFlowMap[humanReadableFlowMap.length - 1].uri.split('/').pop();
      
                // Look for an existing entry with the same result index (using 0-based index)
                let resultEntry = flowMapsByCWE[cwe].find(entry => entry.resultIndex === i);
                if (!resultEntry) {
                  resultEntry = {
                    resultIndex: i,
                    fileName: fileName,
                    flows: []
                  };
                  flowMapsByCWE[cwe].push(resultEntry);
                }
      
                // Append the new flow to the flows array
                resultEntry.flows.push({
                  codeFlowIndex: j,
                  flow: humanReadableFlowMap
                });
              }
            }
            // If there are no codeFlows, we can still build a flow map from locations
            else {
            const flowMap = await this.buildDataFlowMap(result.locations, project);
  
            // Map each node to include additional properties for labeling
            const humanReadable = Object.entries(flowMap).map(([index, node]) => ({
              // Assign unique indexes; you may choose a different strategy if needed
              flowIndex: Number(index),
              vulnerabilityIndex: i,
              step: Number(index), // 0-based step number
              variableName: node.message,
              startLine: node.startLine,
              startColumn: node.startColumn,
              endLine: node.endLine,
              endColumn: node.endColumn,
              uri: node.uri,
              type: node.type,
              code: node.code
            }));
  
            const fileName = humanReadable[humanReadable.length - 1].uri.split('/').pop() ?? 'unknown';
  
            if (!flowMapsByCWE[cwe]) flowMapsByCWE[cwe] = [];
  
            let entry = flowMapsByCWE[cwe].find(e => e.resultIndex === i);
            if (!entry) {
              entry = { resultIndex: i, fileName, flows: [] };
              flowMapsByCWE[cwe].push(entry);
            }
  
            entry.flows.push({ codeFlowIndex: 0, flow: humanReadable });

              }
          }

          // Save the grouped flow maps in human-readable format
          const outputFilePath = path.join(project, 'flowMapsByCWE.json');
          await this.fileService.writeToFile(outputFilePath, JSON.stringify(flowMapsByCWE, null, 2));
      
        } catch (error) {
          console.error('Error processing code flows:', error);
        }
      }
      
    
      /**
       * This function is used to build a map of flow nodes for a given code flow.
       * It reads the file at the specified path and extracts the relevant lines of code.
       * @param codeFlows - The code flows to be processed.
       * @param project - The path to the source directory.
       * @returns - A map of flow nodes with their details.
       */
    async buildDataFlowMap(codeFlows: any[], project: string): Promise<{ [key: number]: FlowNode }> {
        const flowMap: { [key: number]: FlowNode } = {};

        // Use a loop with `await` to ensure file loading is awaited for each code flow
        for (let flowIndex = 0; flowIndex < codeFlows.length; flowIndex++) {
            const locationObj = codeFlows[flowIndex].location ?? codeFlows[flowIndex];
            // console.log(`Processing Code Flow #${flowIndex + 1}`);


            if (locationObj) {
                const physicalLocation = locationObj.physicalLocation ?? {};
                const artifactLocation = physicalLocation.artifactLocation ?? {};
                const uri = artifactLocation.uri ?? 'unknown';
            
                const region = physicalLocation.region ?? {};
                const startLine = region?.startLine ?? NaN;
                const startColumn = region?.startColumn ?? 0;
                const endColumn = region?.endColumn ?? startColumn;
                const endLine = region?.endLine ?? startLine;


                // Normalize the file path for the current OS
                const normalizedUri = path.normalize(decodeURIComponent(uri));
                const fullPath = path.join(project, normalizedUri);

                let message = locationObj.message?.text ?? '';  // Default message from SARIF file
                const type = message.length > 1 ? message.split(':').slice(1).join(':').trim() : '';
                // console.log(`Message: ${message}`);

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
                        uri: decodeURIComponent(uri),  // Keep the original URI for reference
                        startLine: startLine,
                        startColumn: startColumn,
                        endColumn: endColumn,
                        endLine: endLine,
                        type: type,
                        code: codeContext  // Add surrounding code as the context
                    };

                } catch (error) {
                    // console.error(`Error reading file ${fullPath}:`, error);
                    // Optionally handle the error here
                }
            }
        }

        return flowMap; // Return the flow map after all files have been read
    }

    /**
     * This function is used to parse the rules and results from the SARIF file.
     * @param rules - The rules from the SARIF file.
     * @param results - The results from the SARIF file.
     * @param sourcePath - The path to the source directory.
     * @returns - A map of rules with their details and associated files.
     */
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
                    
                    // Here I check for duplicates based on the fileIdentifier. 
                    
                    // It is commented out right now to allign the same number of flows with the map, but eventually it should be used for the last step.
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


    /**
     * This function is used to convert the SARIF results to CSV format.
     * This is used by the PIQUE tool to read the results.
     * @param sourcePath - The path to the source directory.
     * @returns - The CSV data as a string.
     */
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
            let results = fileData.runs[0]?.results || [];

            results = this.filterResults(results)

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
    
    /**
     * This function is used to parse the SARIF results and extract the relevant information.
     * @param rules - The rules from the SARIF file.
     * @param results - The results from the SARIF file.
     * @param sourcePath - The path to the source directory.
     * @returns - A map of rules with their details and associated files.
     */
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

    /**
     * This function is used to correct the path for the current operating system.
     * @param filePath - The path to the source directory.
     * @returns - The corrected path for the current operating system.
     */
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