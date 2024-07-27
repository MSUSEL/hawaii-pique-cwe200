import { Module } from '@nestjs/common';
import { EventsModule } from 'src/events/events.module';
import { BertService } from './bert.service';
import { BertController } from './bert.controller';


@Module({
    imports:[EventsModule],
    controllers: [BertController],
    providers: [BertService],
    exports:[BertService]
})
export class BertModule {}
