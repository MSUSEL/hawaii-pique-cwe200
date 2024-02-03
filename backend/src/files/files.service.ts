import { Global, Injectable } from '@nestjs/common';

import { FileUtilService } from './fileUtilService';
import * as path from 'path';
import { ConfigService } from '@nestjs/config';
import { EventsGateway } from 'src/events/events.gateway';
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
     *
     * @param file Zip binary containing Java Code
     */
    async create(file: Express.Multer.File) {
        // Save file contents to file
        await this.fileUtilService.writeZipFile(file);
        this.eventsGateway.emitDataToClients('data','file uploaded successfully')

        // commenting because unsure if specifically done this way
        // const uploadedProjectDir = path.join(this.projectsPath, file.originalname, 'src');
        const uploadedProjectDir = path.join(this.projectsPath);

        // Get list of java files
        // todo purpose?
        const javaFiles = await this.fileUtilService.getJavaFilesInDirectory(uploadedProjectDir);

        // Return mapped directory tree
        return this.fileUtilService.getDirectoryTree(uploadedProjectDir)
    }

    async findOne(fileId:string) {
        var fileContents=await this.fileUtilService.readFileAsync(fileId)
        return {code:fileContents}
    }
}
