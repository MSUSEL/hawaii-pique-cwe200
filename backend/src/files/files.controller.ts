import {
    Controller,
    Get,
    Post,
    Body,
    Patch,
    UseInterceptors,
    UploadedFile,
    Param,
    Delete,
} from '@nestjs/common';
import { FilesService } from './files.service';
import { FileInterceptor } from '@nestjs/platform-express';
import { AnyFilesInterceptor } from '@nestjs/platform-express';

@Controller('files')
export class FilesController {
    constructor(private readonly filesService: FilesService) {}

    /**
     * Upload Zip File with Multipart file
     *
     * @param file Zip binary containing Java Code
     */
    @Post()
    @UseInterceptors(FileInterceptor('file'))
    create(@UploadedFile() project: Express.Multer.File) {
        return this.filesService.create(project);
    }


    @Post('filecontents')
    findOne(@Body() file:any) {
        return this.filesService.findOne(file.filePath);
    }

}
