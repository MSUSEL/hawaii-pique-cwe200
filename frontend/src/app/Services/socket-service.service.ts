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
    private progressEstimate = new Subject<number>();
    private GPTProgressVariables = new Subject<number>();
    private GPTProgressStrings = new Subject<number>();
    private GPTProgressSinks = new Subject<number>();
    private GPTProgressComments = new Subject<number>();
    private BuildProgress = new Subject<number>();
    private CodeQLProgress = new Subject<number>();


    constructor() { }

    async socketConnect() {
        this.socket = io(this.url);
        this.socket.on('connect', () => {
            console.log('socket connected');
        });

        this.socket.on('data', (data: any) => {
            console.log(data);
            if (this.TerminalStreamingMode) {
                const parsedData = JSON.parse(data);
                // Progress bar for pre-processing files
                if (parsedData.type === 'parsingProgress') {
                    this.progressSubject.next(parsedData.parsingProgress);
                }
                // Progress bar for cost estimate
                else if (parsedData.type === 'estimateProgress') {
                    this.progressEstimate.next(parsedData.estimateProgress);
                }
                else if (parsedData.type === 'GPTProgress-variables') {
                    this.GPTProgressVariables.next(parsedData.GPTProgress);
                }
                else if (parsedData.type === 'GPTProgress-strings') {
                    this.GPTProgressStrings.next(parsedData.GPTProgress);
                }
                else if (parsedData.type === 'GPTProgress-comments') {
                    this.GPTProgressComments.next(parsedData.GPTProgress);
                }
                else if (parsedData.type === 'GPTProgress-sinks') {
                    this.GPTProgressSinks.next(parsedData.GPTProgress);
                }
                else if (parsedData.type === 'CodeQLProgress') {
                    this.CodeQLProgress.next(parsedData.GPTProgress);
                }
                else if (parsedData.type === 'BuildProgress') {
                    this.BuildProgress.next(parsedData.GPTProgress);
                }

                else {
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
    getProgressEstimate(): Observable<number> {
        return this.progressEstimate.asObservable();
    }
    getGPTProgressVariables(): Observable<number> {
        return this.GPTProgressVariables.asObservable();
    }
    getGPTProgressStrings(): Observable<number> {
        return this.GPTProgressStrings.asObservable();
    }
    getGPTProgressComments(): Observable<number> {
        return this.GPTProgressComments.asObservable();
    }
    getGPTProgressSinks(): Observable<number> {
        return this.GPTProgressSinks.asObservable();
    }
    getCodeQLProgress(): Observable<number> {
        return this.CodeQLProgress.asObservable();
    }

    async socketDisconnect() {
        this.socket.off('data');
        if (this.socket.connected) this.socket.disconnect();
    }

    clearOutput() {
        this.socketOutput = [];
    }
}
