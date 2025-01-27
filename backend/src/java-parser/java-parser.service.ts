import { Global, Injectable } from '@nestjs/common';

import * as path from 'path';
import { ConfigService } from '@nestjs/config';
import { EventsGateway } from 'src/events/events.gateway';
import { FileUtilService } from 'src/files/fileUtilService';

@Global()
@Injectable()
export class JavaParserService {
    projectsPath:string;
    parsedResults: { [key: string]: JavaParseResult };

    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
        ){
            this.projectsPath=this.configService.get<string>('CODEQL_PROJECTS_DIR')
            
        }
    
    async getParsedResults(filePaths: string[]) {

            const total = filePaths.length;
            let completed = 0;
            let previousProgress = 0;
        
            // Limit the number of concurrent tasks
            const concurrencyLimit = 8; // Adjust based on your system's capabilities
            const queue = filePaths.slice(); // Create a copy of the file paths
            const results: { [key: string]: JavaParseResult } = {};
        
            const workers = Array.from({ length: concurrencyLimit }, async () => {
                while (queue.length > 0) {
                    const filePath = queue.shift();
                    if (!filePath) break;
        
                    const result = await this.fileUtilService.parseJavaFile(filePath);
                    results[path.basename(result.filename)] = result;
        
                    completed += 1;
                    
                    const progressPercent = Math.floor((completed / total) * 100);
                    if (progressPercent !== previousProgress) {
                        previousProgress = progressPercent;
                        console.log(`Parsing progress: ${progressPercent}%`);
                        this.eventsGateway.emitDataToClients('parsingProgress', JSON.stringify({ type: 'parsingProgress', parsingProgress: progressPercent }));
                    }
                    // console.log(`Parsing progress: ${progressPercent}%`);
                    // this.eventsGateway.emitDataToClients('parsingProgress', JSON.stringify({ type: 'parsingProgress', parsingProgress: progressPercent }));
                }
            });
        
            await Promise.all(workers);
        
            // Assign the aggregated results
            return results;
        }
    }

    interface JavaParseResult {
        filename: string;
        variables: string[];
        comments: string[];
        strings: string[];
        sinks: string[];
    }
    