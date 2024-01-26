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
    async writeZipFile(file: Express.Multer.File) {
        var zip = new AdmZip(file.buffer);
        zip.extractAllTo(
            this.configService.get<string>('CODEQL_PROJECTS_DIR'),
            true,
        );
    }

    async getJavaFilesInDirectory(directoryPath) {
        const files = fs.readdirSync(directoryPath);
        let javaFiles = [];
        for (var file of files) {
            const filePath = path.join(directoryPath, file);
            const fileStats = fs.statSync(filePath);
            if (fileStats.isDirectory()) {
                var newFiles = await this.getJavaFilesInDirectory(filePath);
                javaFiles = javaFiles.concat(newFiles);
            } else if (fileStats.isFile() && file.endsWith('.java')) {
                javaFiles.push(filePath);
            }
        }
        return javaFiles;
    }

    async getDirectoryTree(dirPath: string) {
        const result = [];
        const items = fs.readdirSync(dirPath);
        for (const item of items) {
            const fullPath = path.join(dirPath, item);
            const stat = fs.lstatSync(fullPath);
            if (stat.isDirectory()) {
                let children = await this.getDirectoryTree(fullPath);
                result.push({
                    name: item,
                    type: 'folder',
                    fullPath,
                    children: children,
                });
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
        let concatnatedBatch: string = '';

        const batchResults = await this.processFilesInBatch(batch);

        batchResults.forEach((result) => {
            concatnatedBatch += result;
        });

        // console.log(concatnatedBatch);
        return concatnatedBatch;
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

        let processedFile: string = '';
        processedFile += '-----BEGIN FILE: [' + filePath + ']-----';
        let inMultilineComment = false;

        for await (const line of rl) {
            let trimmedLine = line.trim();
            if (trimmedLine.startsWith('/*')) {
                inMultilineComment = true;
            } else if (trimmedLine.endsWith('*/')) {
                inMultilineComment = false;
            } else if (
                !inMultilineComment &&
                !trimmedLine.startsWith('//') &&
                trimmedLine !== '' &&
                !trimmedLine.startsWith('import')
            ) {
                processedFile += trimmedLine;
            }
        }
        processedFile += '-----END FILE: [' + filePath + ']-----';
        return processedFile;
    }
}
