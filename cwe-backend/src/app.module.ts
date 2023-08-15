import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { FilesModule } from './files/files.module';
import { ChatGptModule } from './chat-gpt/chat-gpt.module';
import { ConfigModule } from '@nestjs/config';
import { CodeQlModule } from './code-ql/code-ql.module';
@Module({
    imports: [
        ConfigModule.forRoot({
            envFilePath: '.env',
            isGlobal: true,
        }),
        FilesModule,
        ChatGptModule,
        CodeQlModule,
    ],
    controllers: [AppController],
    providers: [AppService],
})
export class AppModule {}
