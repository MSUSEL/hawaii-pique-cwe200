import { Injectable } from '@angular/core';
import {
    HttpClient,
    HttpErrorResponse,
    HttpEvent,
    HttpRequest,
} from '@angular/common/http';
import { catchError, Observable, of } from 'rxjs';
import { environment } from '../../environments/environment';
@Injectable({
    providedIn: 'root',
})
export class FilesService {
    public url = environment.apiUrl + '/files/';
    constructor(private http: HttpClient) {}

    uploadFile(data: any): Observable<any> {
        return this.http.post(this.url, data);
    }
    getFile(id: string) {
        this.http.get(this.url + 'file/' + id);
    }

    getFileContents(filePath:any):Observable<any>{
        return this.http.post(this.url+"filecontents",filePath);
    }
    getFileUrl(id: string) {
        return this.url + 'file/' + id;
    }

    getFileAttachment(id: string): Observable<any> {
        return this.http.get(this.url + 'file/' + id, {
            responseType: 'blob',
            reportProgress: true,
        });
    }

    delete(id: string): Observable<any> {
        return this.http.delete(this.url + id);
    }
}
