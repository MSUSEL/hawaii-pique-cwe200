import { Global, Injectable } from '@nestjs/common';

import { FileUtilService } from './fileUtilService';
import * as path from 'path';
import { ConfigService } from '@nestjs/config';
@Global()
@Injectable()
export class FilesService {
    projectsPath:string;
    constructor(
        private configService: ConfigService,
        private fileUtilService:FileUtilService){
            this.projectsPath=this.configService.get<string>('CODEQL_PROJECTS_DIR')
        }
    async create(file: Express.Multer.File) {
        await this.fileUtilService.writeZipFile(file);
        
        var uploadedProjectDir=path.join(this.projectsPath,file.originalname,'src')
        var javaFiles=await this.fileUtilService.getJavaFilesInDirectory(uploadedProjectDir);
        return this.fileUtilService.getDirectoryTree(uploadedProjectDir)
    }

    async findOne(fileId:string) {
        var fileContents=await this.fileUtilService.readFileAsync(fileId)
        return {code:fileContents}
    }
}
