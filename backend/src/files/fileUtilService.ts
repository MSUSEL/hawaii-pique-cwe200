import {Global, Injectable} from '@nestjs/common';
import {ConfigService} from '@nestjs/config';
import { exec } from 'child_process';
import * as AdmZip from 'adm-zip';
import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';

@Global()
@Injectable()
export class FileUtilService {
    constructor(private configService: ConfigService) {}

    /**
     * Extract Zip contents to file
     *
     * @param file Zip binary containing Java Code
     */
    async writeZipFile(file: Express.Multer.File) {
        // Create new unzipper
        const zip = new AdmZip(file.buffer);

        // Create new project directory if DNE
        // TODO create if DNE during launch?
        if (!fs.existsSync(this.configService.get<string>('CODEQL_PROJECTS_DIR')))
            fs.mkdirSync(this.configService.get<string>('CODEQL_PROJECTS_DIR'));

        // Extract code to project directory
        zip.extractAllTo(
            this.configService.get<string>('CODEQL_PROJECTS_DIR') + "/" + file.originalname,  // extract name
            true    // overwrite
        );
    }

    /**
     * Recursively find all files with a .java extension in a given directory
     *
     * @param directoryPath Path to root directory to search for files
     */
    async getJavaFilesInDirectory(directoryPath) {
        // Get all files/dir in pwd
        const files = fs.readdirSync(directoryPath);
        let javaFiles = [];

        // Iterate through directory contents
        for (const file of files) {

            // Get file details
            const filePath = path.join(directoryPath, file);
            const fileStats = fs.statSync(filePath);

            // If directory, open
            if (fileStats.isDirectory()) {
                const newFiles = await this.getJavaFilesInDirectory(filePath);
                javaFiles = javaFiles.concat(newFiles);
            // else get file information
            } else if (fileStats.isFile() && file.endsWith('.java')) {
                javaFiles.push(filePath);
            }
        }
        // Return list of java files
        return javaFiles;
    }


    /**
     * Recursively build a list of files in a given directory
     *
     * @param dirPath Root to build tree from
     */
    async getDirectoryTree(dirPath: string) {
        const result = [];

        // Iterate over each item in
        for (const item of fs.readdirSync(dirPath)) {

            // Get file details
            const fullPath = path.join(dirPath, item);
            const stat = fs.lstatSync(fullPath);

            // If directory, get children and add
            if (stat.isDirectory() ) {
                let children = await this.getDirectoryTree(fullPath);
                result.push({
                    name: item,
                    type: 'folder',
                    fullPath,
                    children: children,
                });
            // Else push the file
            } else {
                result.push({
                    name: item,
                    type: 'file',
                    fullPath,
                    children: null,
                });
            }
        }
        return result;
    }

    async readFileAsync(filePath: string) {
        if (fs.existsSync(filePath)) {
            try {
                const code = fs.readFileSync(filePath, 'utf8');
                return code;
            } catch (error) {
                console.error(error);
                return '';
            }
        }
    }

    readJsonFile(filePath: string) {
        if (fs.existsSync(filePath)) {
            try {
                const data = fs.readFileSync(filePath, 'utf8');
                return JSON.parse(data);
            } catch (error) {
                console.error(error);
                return '';
            }
        }
    }

    parseJSONFile(filePath: string) {
        
        let variables = [];
        const fileList: any[] = [];
        let comments = [];
        
        let sensitiveVariablesMapping = new Map<string, string[]>();
        let sensitiveStringsMapping = new Map<string, string[]>();
        let sensitiveCommentsMapping = new Map<string, string[]>();

        if (fs.existsSync(filePath)) {
            try {
                const data = fs.readFileSync(filePath, 'utf8');
                const json = JSON.parse(data);

            json.forEach((file) => {
                let key = file["fileName"];
                let sensitiveVariables = file["sensitiveVariables"].map((variable) => `"${variable["name"]}"`);
                let sensitiveStrings = file["sensitiveStrings"].map((str) => `"${str["name"]}"`);
                let sensitiveComments = file["sensitiveComments"].map((comment) => `"${comment["name"]}"`);

                fileList.push({
                    fileName: key,
                    sensitiveVariables: sensitiveVariables,
                    sensitiveStrings: sensitiveStrings,
                    sensitiveComments: sensitiveComments
                });

                variables = variables.concat(sensitiveVariables);
                sensitiveVariablesMapping[key] = sensitiveVariables;
                sensitiveStringsMapping[key] = sensitiveStrings;
                sensitiveCommentsMapping[key] = sensitiveComments;
                comments = comments.concat(sensitiveComments);

            });

            } catch (error) {
                console.error(error);
            }
        }

        return { variables, fileList, sensitiveVariablesMapping, sensitiveStringsMapping, comments};

    }

    /**
     * Remove directory and contents if it exists
     *
     * @param dirPath path to directory to delete
     */
    async removeDir(dirPath: string) {
        if (fs.existsSync(dirPath))
            fs.rmSync(dirPath, { recursive: true });
    }

    processFilePath(sourcePath: string, filePath: string) {
        return path.join(sourcePath, filePath.replace('/', '\\'));
    }
    getFilenameFromPath(filePath: string) {
        var index = filePath.lastIndexOf('/');
        return filePath.substring(index + 1);
    }

    /**
     * Write to path with utf-8 encoding
     *
     * @param filePath path to write to
     * @param content content of file
     */
    async writeToFile(filePath: string, content: string) {
        fs.writeFileSync(filePath, content, 'utf-8');
    }


    /**
     * Process single java file by replacing class name with ID, and removing blank lines
     * @param filePath Path to java file to process
     * @param id ID to replace class name with
     */
    async processJavaFile(filePath: string, id: string): Promise<string> {
        let baseName = path.basename(filePath).split('.java')[0];
        // Read file contents
        const fileStream = fs.createReadStream(filePath);
        const rl = readline.createInterface({
            input: fileStream,
            crlfDelay: Infinity,
        });

        // Initialize new processed file
        let processedLines: string[] = [];
        let classNameChanged = false;
        let lines = [];
        for await (const line of rl) {
            lines.push(line);
        }

        for (let i = 0; i < lines.length; i++) {
            let trimmedLine = lines[i].replace(/^\s*[\r\n]/gm, '');

            // Check if the line contains the class declaration
            if (!classNameChanged && trimmedLine.includes('class ') && trimmedLine.includes(baseName)) {
                // Replace the class name with the given ID
                trimmedLine = trimmedLine.replace(baseName, id);
            }

            // Check if the next line contains only }, ), or ;
            if (i < lines.length - 1 && /^[\}\);\s]*$/.test(lines[i + 1])) {
                trimmedLine += ' ' + lines[i + 1].trim();
                i++; // Skip next iteration
            }

            // Skip blank lines
            if (trimmedLine.trim() !== '') {
                processedLines.push(trimmedLine);
            }
        }

        // Return processed file content as a string
        return processedLines.join('\n');
    }   
    

    addFileBoundaryMarkers(id: string, file: string){
        // let fileName = path.basename(filePath);
        return '\n\n-----BEGIN FILE: [' + id + ']----- \n' + file + '\n-----END FILE: [' + id + ']-----'

    }
        
    parseJavaFile(filePath: string, aggregatedResults: { [key: string]: JavaParseResult }): Promise<void> {
        return new Promise((resolve, reject) => {
            const cwd = process.cwd();
            const jarPath = path.resolve(cwd, 'ParseJava', 'target', 'ParseJava-1.0-jar-with-dependencies.jar');
            filePath = path.resolve(cwd, filePath);
    
            if (!fs.existsSync(jarPath)) {
                exec('mvn clean package', { cwd: path.resolve(cwd, 'ParseJava') }, (error, stdout, stderr) => {
                    if (error) {
                        reject(`Error building JAR: ${stderr}`);
                        return;
                    }
                    this.runJavaProgram(jarPath, filePath, aggregatedResults).then(resolve).catch(reject);
                });
            } else {
                this.runJavaProgram(jarPath, filePath, aggregatedResults).then(resolve).catch(reject);
            }
        });
    }
    
    runJavaProgram(jarPath: string, filePath: string, aggregatedResults: { [key: string]: JavaParseResult }): Promise<void> {
        return new Promise((resolve, reject) => {
            const command = `java -jar ${jarPath} ${filePath}`;
    
            exec(command, (error, stdout, stderr) => {
                if (error) {
                    console.error(`Error executing command for file ${filePath}: ${stderr}`);
                    // Add empty result to aggregated results
                    aggregatedResults[path.basename(filePath)] = {
                        filename: path.basename(filePath),
                        variables: [],
                        comments: [],
                        strings: [],
                    };
                    resolve();
                    return;
                }
    
                try {
                    const result: JavaParseResult = JSON.parse(stdout);
                    aggregatedResults[path.basename(result.filename)] = result;
                    resolve();
                } catch (e) {
                    console.error(`Failed to parse JSON: ${e}`);
                    aggregatedResults[path.basename(filePath)] = {
                        filename: path.basename(filePath),
                        variables: [],
                        comments: [],
                        strings: [],
                    };
                    resolve();
                }
            });
        });
    }
  
}

interface JavaParseResult {
    filename: string;
    variables: string[];
    comments: string[];
    strings: string[];
}

