import { HttpClient, HttpParams  } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from 'src/environments/environment';


@Injectable({
    providedIn: 'root',
})
export class CodeQlService {
    public url = environment.apiUrl + '/codeql/';
    constructor(private http: HttpClient) {}

    runCodeQl(data:any):Observable<any>{
        return this.http.post(this.url,data);
    }
    
    getSarifResult(project : string):Observable<any>{
        const params = new HttpParams().set('project', project);
        return this.http.get(this.url, { params });
    }
}
