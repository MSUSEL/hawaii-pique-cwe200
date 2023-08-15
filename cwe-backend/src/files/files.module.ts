import { Global, Module } from '@nestjs/common';
import { FilesService } from './files.service';
import { FilesController } from './files.controller';
import { FileUtilService } from './fileUtilService';
@Global()
@Module({
    controllers: [FilesController],
    providers: [FilesService, FileUtilService],
    exports: [FileUtilService],
})
export class FilesModule {}
