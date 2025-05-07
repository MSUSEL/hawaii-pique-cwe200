import {
    SubscribeMessage,
    WebSocketGateway,
    WebSocketServer,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import { Injectable } from '@nestjs/common';

/**
 * EventsGateway is a WebSocket gateway that handles real-time events.
 * It listens for incoming messages and emits data to connected clients.
 */
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

    emitDataToClients(type: string, data: string) {
        this.server.emit('data', data);  // Ensure this emits on the 'data' event
    }
}
