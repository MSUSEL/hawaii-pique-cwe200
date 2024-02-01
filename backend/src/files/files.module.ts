import { Global, Module } from '@nestjs/common';
import { FilesService } from './files.service';
import { FilesController } from './files.controller';
import { FileUtilService } from './fileUtilService';
import { EventsModule } from 'src/events/events.module';
@Global()
@Module({
    imports:[EventsModule],
    controllers: [FilesController],
    providers: [FilesService, FileUtilService],
    exports: [FileUtilService],
})
export class FilesModule {}
