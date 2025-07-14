import {
    Controller
} 
from '@nestjs/common';
import { JavaParserService } from './implementations/java-parser.service';
import { FileInterceptor } from '@nestjs/platform-express';
import { AnyFilesInterceptor } from '@nestjs/platform-express';

@Controller('java-parser')
export class JavaParserController {
    constructor(private readonly javaparserservice: JavaParserService) {}


}
