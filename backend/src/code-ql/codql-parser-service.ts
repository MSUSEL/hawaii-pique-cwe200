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

    parseRules(rules: any[], results: any[],sourcePath:string) {
        var rulesList = [];
        for (let i = 0; i < rules.length; i++) {
            let rule = rules[i];
            var files = results
                .filter((item) => item.ruleIndex == i)
                .map((file) => ({
                    name: this.fileService.getFilenameFromPath(
                        file.locations[0]?.physicalLocation.artifactLocation
                            .uri,
                    ),
                    fullPath: this.fileService.processFilePath(sourcePath,
                        file.locations[0]?.physicalLocation.artifactLocation
                            .uri,
                    ),
                    message: file.message.text,
                    region: file.locations[0]?.physicalLocation.region,
                }));

                var object={
                    name: rule.id,
                    type: rule.defaultConfiguration.level,
                    message: rule.fullDescription
                        ? rule.fullDescription.text
                        : rule.shortDescription.text,
                    files: files,
                }
                if(object.files.length) rulesList.push(object);
        }
        return rulesList;
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
