import { HttpClient, HttpParams  } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';


@Injectable({
    providedIn: 'root',
})
export class AnalyzeService {
    public url = environment.apiUrl + '/analyze/';
    constructor(private http: HttpClient) {}

    analyze(data:any):Observable<any>{
        return this.http.post(this.url, data); 
    }
}
