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
import { JavaParserService } from './java-parser.service';
import { FileInterceptor } from '@nestjs/platform-express';
import { AnyFilesInterceptor } from '@nestjs/platform-express';

@Controller('java-parser')
export class JavaParserController {
    constructor(private readonly javaparserservice: JavaParserService) {}


}
