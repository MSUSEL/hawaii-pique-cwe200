import { Injectable } from '@nestjs/common';
import { FileUtilService } from 'src/files/fileUtilService';
import * as path from 'path'

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

    parseRules(rules: any[], results: any[], sourcePath: string) {
        const rulesMap = new Map();
        
        for (let i = 0; i < rules.length; i++) {
            let rule = rules[i];
            let ruleDescription = cweDescriptions.get(rule.id) || "Description not available";
            let ruleKey = `${rule.id}: ${ruleDescription}`; // Concatenate rule ID and description
    
            var files = results
                .filter((item) => item.ruleIndex == i)
                .map((file) => ({
                    name: this.fileService.getFilenameFromPath(
                        file.locations[0]?.physicalLocation.artifactLocation.uri,
                    ),
                    fullPath: this.fileService.processFilePath(
                        sourcePath,
                        file.locations[0]?.physicalLocation.artifactLocation.uri,
                    ),
                    message: file.message.text,
                    region: file.locations[0]?.physicalLocation.region,
                }));
    
            // Check if the concatenated ruleKey already exists in the map
            if (rulesMap.has(ruleKey)) {
                const existingEntry = rulesMap.get(ruleKey);
                existingEntry.files = existingEntry.files.concat(files);
                rulesMap.set(ruleKey, existingEntry);
            } else {
                // Create a new entry with the concatenated ruleKey if it does not exist
                var object = {
                    name: ruleKey,
                    type: rule.defaultConfiguration.level,
                    message: rule.fullDescription ? rule.fullDescription.text : rule.shortDescription.text,
                    files: files,
                    description: ruleDescription,
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
            var fullPath = this.fileService.processFilePath(sourcePath,filePath);
            var fileIndex = resultList.findIndex(
                (file) => file.fullPath == fullPath,
            );
            var file: any = null;
            if (fileIndex == -1) {
                file = {
                    name: this.fileService.getFilenameFromPath(filePath),
                    fullPath: this.fileService.processFilePath(sourcePath,filePath),
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
