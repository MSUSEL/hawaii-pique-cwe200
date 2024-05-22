import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';

@Injectable({
    providedIn: 'root',
})
export class ChatGptService {
    public url = environment.apiUrl + '/chatgpt/';
    public ChatGPTToken = null;
    public GPTModel = null;

    constructor(private http: HttpClient) {}

    queryChatGpt(data:any):Observable<any>{
        return this.http.post(this.url,data);
    }

    getCostEstimate(projectPath:string):Observable<any>{
        return this.http.get(this.url+'?project='+projectPath);
    }

    getChatGptToken():Observable<any>{
        return this.http.get(this.url+'token');
    }

    updateChatGptToken(token:string):Observable<any>{
        return this.http.post(this.url+'token',{token:token});
    }
}
