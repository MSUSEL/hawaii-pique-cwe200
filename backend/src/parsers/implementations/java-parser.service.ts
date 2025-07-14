import { Global, Injectable } from '@nestjs/common';

import * as path from 'path';
import * as fs from 'fs';
import { exec } from 'child_process';
import { ConfigService } from '@nestjs/config';
import { EventsGateway } from 'src/events/events.gateway';
import { FileUtilService } from 'src/files/fileUtilService';
import { ParserBase, ParseResult } from '../base/parser-base';

/**
 * JavaParserService is a service that provides functionality to parse Java files and extract relevant information.
 */
@Global()
@Injectable()
export class JavaParserService extends ParserBase {
    projectsPath: string;
    parsedResults: { [key: string]: ParseResult };
    projectPath: string;

    constructor(
        private configService: ConfigService,
        eventsGateway: EventsGateway,
        private fileUtilService: FileUtilService,
    ) {
        super(eventsGateway);
        this.projectsPath = this.configService.get<string>('CODEQL_PROJECTS_DIR')

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
        await this.buildJarIfNeeded();
        this.parsedResults = await this.getParsedResults(filePaths);
        await this.fileUtilService.writeToFile(path.join(this.projectPath, 'parsedResults.json'), JSON.stringify(this.parsedResults, null, 2));
    }

    /**
     * Builds the "ParseJava" JAR file if it doesn't exist.
     * @returns A promise that resolves when the JAR file is built.
     **/
    async buildJarIfNeeded() {
        const cwd = process.cwd();
        const jarPath = path.resolve(cwd, 'ParseJava', 'target', 'ParseJava-1.0-jar-with-dependencies.jar');

        if (!fs.existsSync(jarPath)) {
            // Build the JAR file
            console.log("Building JAR for Java Parser")
            await new Promise<void>((resolve, reject) => {
                exec('mvn clean package', { cwd: path.resolve(cwd, 'ParseJava') }, (error, stdout, stderr) => {
                    if (error) {
                        reject(`Error building JAR: ${stderr}`);
                        return;
                    }
                    resolve();
                });
            });
        }
    }

    /**
     * Wrapper for calling the JAR in "backend/ParseJava/target/ParseJava-1.0-jar-with-dependencies.jar".
     * @param filePath - The path to the Java file to be parsed.
     * @returns A promise that resolves to an object containing the parsed results.
     **/
    async parseSourceFile(filePath: string): Promise<ParseResult> {
        const cwd = process.cwd();
        const jarPath = path.resolve(cwd, 'ParseJava', 'target', 'ParseJava-1.0-jar-with-dependencies.jar');
        filePath = path.resolve(cwd, filePath);

        // Run the Java program
        const command = `java -jar ${jarPath} ${filePath}`;
        const result = await this.runExternalParser(command, filePath);
        return result;
    }

}

