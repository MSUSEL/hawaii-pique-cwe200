import { Global, Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
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
            this.configService.get<string>('CODEQL_PROJECTS_DIR') + "/" + file.originalname.split(".")[0],  // extract name
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

    async removeDir(dirPath: string) {
        if (fs.existsSync(dirPath)) {
            fs.rmSync(dirPath, { recursive: true });
        }
    }

    processFilePath(sourcePath: string, filePath: string) {
        return path.join(sourcePath, filePath.replace('/', '\\'));
    }
    getFilenameFromPath(filePath: string) {
        var index = filePath.lastIndexOf('/');
        return filePath.substring(index + 1);
    }

    async writeToFile(filePath: string, content: string) {
        fs.writeFileSync(filePath, content, 'utf-8');
    }

    // Preprocesses files by removing comments and imports, and concatenating multiple of them into a single string
    async preprocessFiles(batch: string[]) {

        const batchResults = await this.processFilesInBatch(batch);
        const concatenatedBatch = batchResults.join('');

        // console.log(concatnatedBatch);
        return concatenatedBatch;
    }

    async processFilesInBatch(filePaths: string[]): Promise<string[]> {
        return Promise.all(
            filePaths.map((filePath) => this.processJavaFile(filePath)),
        );
    }

    async processJavaFile(filePath: string): Promise<string> {
        const fileStream = fs.createReadStream(filePath);
        const rl = readline.createInterface({
            input: fileStream,
            crlfDelay: Infinity,
        });
    
        let processedLines: string[] = [];
        processedLines.push('-----BEGIN FILE: [' + filePath + ']-----');
        let inMultilineComment = false;
        const sensitiveKeywords = [
            "copyright"
        ];
    
        for await (const line of rl) {
            let trimmedLine = line.trim();
            if (trimmedLine.startsWith('/*')) {
                inMultilineComment = true;
            }
    
            if (inMultilineComment || trimmedLine.startsWith('//')) {
                // If the line is a comment, check if it contains a sensitive keyword
                if (sensitiveKeywords.some(keyword => trimmedLine.toLowerCase().includes(keyword))) {
                    processedLines.push(trimmedLine);
                }
            } else if (trimmedLine) {
                processedLines.push(trimmedLine);
            }
    
            if (trimmedLine.endsWith('*/')) {
                inMultilineComment = false;
            }
        }
        processedLines.push('-----END FILE: [' + filePath + ']-----');
    
        return processedLines.join('\n');
    }
    
    
}
