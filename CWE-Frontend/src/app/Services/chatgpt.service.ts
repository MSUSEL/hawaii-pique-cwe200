import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';

@Injectable({
    providedIn: 'root',
})
export class ChatGptService {
    public url = environment.apiUrl + '/chatgpt/';
    constructor(private http: HttpClient) {}

    queryChatGpt(data:any):Observable<any>{
        return this.http.post(this.url,data);
    }
}
