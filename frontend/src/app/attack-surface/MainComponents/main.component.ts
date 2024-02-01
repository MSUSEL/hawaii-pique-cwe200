import { Component, OnInit, Input, ViewEncapsulation } from '@angular/core';
import {  CVEUtilService } from 'src/app/attack-surface/cwe-util.service';

@Component({
    selector: 'app-main-cwe',
    templateUrl: './main.component.html',
    encapsulation: ViewEncapsulation.None
})
export class MainComponent implements OnInit {

    SourceClass:any;
    constructor(
        public utilService: CVEUtilService
    ) {}

    ngOnInit(): void {

    }
    isCompleted(property:any){
        return property.Value.length>0;
    }
    
    isEmpty(arrayProperty:Array<any>){
        return arrayProperty.length==0;
    }
}
