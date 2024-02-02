import { Injectable } from '@angular/core';
import { CdkDragDrop, moveItemInArray } from '@angular/cdk/drag-drop';

import { saveAs } from 'file-saver';


@Injectable({
    providedIn: 'root',
})
export class CVEUtilService {
    SelectedProject:File=new File([],"");
    ProjectName:string="New Project";
    view:string="main";
    public isSubmitted:boolean=false;
    directoryFilesTree:any[]=[];
    isUploaded:boolean=false;
    constructor(


    ) {}

    importFromJson(event: any, inputFile: HTMLInputElement) {
        if (event.target.files.length > 0) {
            this.SelectedProject = <File>event.target.files[0]; 
            var extenstionIndex = this.SelectedProject.name.lastIndexOf('.');
            this.ProjectName = this.SelectedProject.name.substring(
                0,
                extenstionIndex
            );  
            this.isUploaded=false;
            inputFile.value = '';
        }
    }

    newCVE() {
        this.SelectedProject=new File([],"");
        this.ProjectName="New Project";
        this.isSubmitted=false;
        this.isUploaded=false;
        this.view="main"
       
    }
    isValidProject(){
        return this.SelectedProject.name!="";
    }

    readyForUpload(){
        return this.isValidProject() && this.isUploaded==false;
    }
    readyForAnalysis(){
        return this.isValidProject() && this.isUploaded==true;
    }
    export(){    
    }
}
