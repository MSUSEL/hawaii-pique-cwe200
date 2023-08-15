import { Injectable } from "@angular/core";

@Injectable({
    providedIn: "root",
})
export class HttpLoadingService {
    private loading: boolean = false;
    private title:string;
    constructor() {}

    set Title(value:string){
        this.title=value;
    }
    get Title():string{
        return this.title;
    }

    setLoading(loading: boolean) {
        this.loading = loading;
    }

    getLoading(): boolean {
        return this.loading;
    }
}
