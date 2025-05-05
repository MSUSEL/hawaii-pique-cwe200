import { Global, Injectable } from '@nestjs/common';

import { FileUtilService } from './fileUtilService';
import * as path from 'path';
import { ConfigService } from '@nestjs/config';
import { EventsGateway } from 'src/events/events.gateway';

/**
 * * FilesService is a service that provides functionality to handle file operations.
 */
@Global()
@Injectable()
export class FilesService {
    projectsPath:string;
    constructor(
        private configService: ConfigService,
        private eventsGateway: EventsGateway,
        private fileUtilService:FileUtilService){
            this.projectsPath=this.configService.get<string>('CODEQL_PROJECTS_DIR')
        }

    /**
     * Create a directory tree for a given file
     * @param file Zip binary containing Java Code
     */
    async create(file: Express.Multer.File) {
        // Save file contents to file        
        await this.fileUtilService.writeZipFile(file);
        this.eventsGateway.emitDataToClients('data','file uploaded successfully')
        let projectName = file.originalname;
        if (projectName.endsWith('.zip')) {
            projectName = projectName.slice(0, -4); // Remove '.zip'
        }

        const uploadedProjectDir = path.join(this.projectsPath, projectName);

        // Return mapped directory tree
        return this.fileUtilService.getDirectoryTree(uploadedProjectDir)
    }

    /**
     * Get the contents of a file
     * @param fileId The ID of the file to retrieve
     * @returns The contents of the file
     */
    async findOne(fileId:string) {
        var fileContents=await this.fileUtilService.readFileAsync(fileId)
        return {code:fileContents}
    }
}
