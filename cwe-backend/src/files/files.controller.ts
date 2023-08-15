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

    @Post()
    @UseInterceptors(FileInterceptor('file'))
    create(@UploadedFile() file: Express.Multer.File) {
        return this.filesService.create(file);
    }



    @Post('filecontents')
    findOne(@Body() file:any) {
        return this.filesService.findOne(file.filePath);
    }

}
