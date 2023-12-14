import { ValidationPipe } from '@nestjs/common';
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { IoAdapter } from '@nestjs/platform-socket.io';
import { Server } from 'socket.io';

class ExtendedIoAdapter extends IoAdapter {
    createIOServer(port: number, options?: any): any {
        const server: Server = super.createIOServer(port, {
            ...options,
            cors: {
                origin: ['http://129.21.128.6:5300', 'http://localhost:4200'],
                methods: ['GET', 'POST'],
                credentials: true,
            },
        });
        return server;
    }
}

async function bootstrap() {
    const app = await NestFactory.create(AppModule);
    app.useGlobalPipes(
        new ValidationPipe({
            whitelist: true,
        }),
    );

    app.enableCors(
        {
            origin: ['http://129.21.128.6:5300', 'http://localhost:4200'],
        }
    );
    app.useWebSocketAdapter(new ExtendedIoAdapter(app));
    await app.listen(5400);
}
bootstrap();
