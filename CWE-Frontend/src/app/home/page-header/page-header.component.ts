import { Component, OnInit, Input } from '@angular/core';
import { CodeQlService } from 'src/app/Services/codeql-service';
import { FilesService } from 'src/app/Services/fileService';
import { SocketService } from 'src/app/Services/socket-service.service';
import { CVEUtilService } from 'src/app/attack-surface/cwe-util.service';
import { EditorService } from 'src/app/shared-components/editor-service';
declare var $: any;
@Component({
    selector: 'app-page-header',
    templateUrl: './page-header.component.html',
})
export class PageHeaderComponent implements OnInit {
    isLoading:boolean=false;
    constructor(
        public utilService: CVEUtilService,
        private socketService:SocketService,
        private fileService: FilesService,
        private codeQlService:CodeQlService,
        private editorService:EditorService
    ) {}

    ngOnInit(): void {
        $(document).ready(function () {
            $('#right-sidebar-collapse').on('click', function () {
                $('.wrapper').toggleClass('overflow-hidden');
                $('.right-sidebar').toggleClass('active');
                $('#right-sidebar-collapse i').toggleClass(
                    'fa-flip-horizontal'
                );
            });
            $('#left-sidebar-collapse').on('click', function () {
                $('.left-sidebar').toggleClass('active');
                $('#left-sidebar-collapse i').toggleClass('fa-flip-horizontal');
            });
            
            $('#terminal-collapse,#terminal-dismiss').on('click', function () {
                $('#terminal').toggleClass('d-none');
            });
        });
    }

    uploadProject() {
        var data: FormData = new FormData();
        data.append('file', this.utilService.SelectedProject, this.utilService.ProjectName);

        this.fileService.uploadFile(data).subscribe((response)=>{
            this.utilService.directoryFilesTree=response;
            this.utilService.isUploaded=true;
        })
    }


    async runCodeQl(){
        this.socketService.clearOutput();
        await this.socketService.socketConnect();
        this.isLoading=true;
        this.codeQlService.runCodeQl({project:this.utilService.ProjectName}).subscribe((response)=>{
            this.editorService.rulesTree=response.rulesTree;
            this.editorService.locationsTree=response.locationsTree;
            this.isLoading=false;
            this.socketService.socketDisconnect();
        })
    }

}
