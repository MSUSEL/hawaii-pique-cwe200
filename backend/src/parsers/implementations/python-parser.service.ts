import { Global, Injectable } from '@nestjs/common';

import * as path from 'path';
import * as fs from 'fs';
import { exec } from 'child_process';
import { ConfigService } from '@nestjs/config';
import { EventsGateway } from 'src/events/events.gateway';
import { FileUtilService } from 'src/files/fileUtilService';
import { ParserBase, ParseResult } from '../base/parser-base';

/**
 * PythonParserService is a service that provides functionality to parse Python files and extract relevant information.
 */
@Global()
@Injectable()
export class PythonParserService extends ParserBase {
    projectsPath: string;
    parsedResults: { [key: string]: ParseResult };
    projectPath: string;
    command: string;

    constructor(
        private configService: ConfigService,
        eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
    ) {
        super(eventsGateway);
        this.projectsPath = this.configService.get<string>('CODEQL_PROJECTS_DIR');
    }
    /**
     * A wrapper that kicks off the parsing.
     * @param filePaths - An array of file paths to be parsed.
     * @param sourcePath - The path to the source directory.
     * @returns A promise that resolves to an object containing the parsed results.
     * @throws Error if the parsing fails.
     **/
    async wrapper(filePaths: string[], sourcePath: string) {
        this.projectPath = sourcePath;
        this.parsedResults = await this.getParsedResults(filePaths);
        await this.fileUtilService.writeToFile(path.join(this.projectPath, 'parsedResults.json'), JSON.stringify(this.parsedResults, null, 2));
    }
    

    /**
     * Wrapper for calling the Python script in "backend/ParsePython".
     * @param filePath - The path to the Python file to be parsed.
     * @returns A promise that resolves to an object containing the parsed results.
     **/
    async parseSourceFile(filePath: string): Promise<ParseResult> {
        const cwd = process.cwd();
        const pythonParserPath = path.resolve(cwd, 'ParsePython', 'parse_python.py');
        filePath = path.resolve(cwd, filePath);

        // Run the Python program
        const command = `python ${pythonParserPath} "${filePath}"`;
        const result = await this.runExternalParser(command, filePath);
        return result;
    }

}

