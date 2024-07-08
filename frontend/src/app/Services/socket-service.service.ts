import { Injectable } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { environment } from '../../environments/environment';
import { Observable, Subject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class SocketService {
  public socket: Socket;
  public socketOutput: any[] = [];
  TerminalStreamingMode: boolean = true;
  public url = environment.apiUrl;

  private progressSubject = new Subject<number>();

  constructor() {}

  async socketConnect() {
    this.socket = io(this.url);
    this.socket.on('connect', () => {
      console.log('socket connected');
    });

    this.socket.on('data', (data: any) => {
        console.log('Data:', data);
      if (this.TerminalStreamingMode) {
        const parsedData = JSON.parse(data);
        if (parsedData.type === 'progress') {
          this.progressSubject.next(parsedData.progress);
        } else {
          this.socketOutput.push({ type: 'data', data: parsedData });
          const element = document.getElementById('terminal');
          if (element) {
            element.scrollTop = element.scrollHeight;
          }
        }
      }
    });
  }

  getProgressUpdates(): Observable<number> {
    return this.progressSubject.asObservable();
  }

  async socketDisconnect() {
    this.socket.off('data');
    if (this.socket.connected) this.socket.disconnect();
  }

  clearOutput() {
    this.socketOutput = [];
  }
}
