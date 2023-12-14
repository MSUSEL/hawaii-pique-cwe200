import { Injectable } from '@angular/core';
import {io,Socket} from 'socket.io-client';
import { environment } from '../../environments/environment';
import { Observable ,Subject } from 'rxjs';
@Injectable({
    providedIn: 'root',
})
export class SocketService {
    public socket:Socket;
    public socketOutput:any[]=[];
    TerminalStreamingMode:boolean=true;
    public url = environment.apiUrl;
    constructor() {
        
    }
    async socketConnect() {
        this.socket = io(this.url);
        this.socket.on('connect', () => {
            console.log('socket connected');
        });

        this.socket.on('data', (data: any) => {
            if(this.TerminalStreamingMode){
                this.socketOutput.push({type:'data',data})
                var element = document.getElementById("terminal");
                element.scrollTop = element.scrollHeight;
            }
        });
    }

    async socketDisconnect() {
        this.socket.off("data")
        if (this.socket.connected)  this.socket.disconnect();
        
    }

    clearOutput(){
        this.socketOutput=[];
    }
}
