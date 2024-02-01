import {
    SubscribeMessage,
    WebSocketGateway,
    WebSocketServer,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Global, Injectable } from '@nestjs/common';
@Injectable()
@WebSocketGateway()
export class EventsGateway {
    @WebSocketServer() server: Server;
    @SubscribeMessage('message')
    handleMessage(client: Socket, payload: any) {
        this.server.emit('data', 'Hello everyone!');
    }

    handleConnection(client: any, ...args: any[]) {
        console.log('Client connected:', client.id);
    }
    handleDisconnect(client: any) {
        console.log('Client disconnected:', client.id);
    }

    emitDataToClients(type:string,data:string) {
        this.server.emit(type, data);
    }
}
