import { Component, OnInit, Input } from '@angular/core';
import { CodeQlService } from 'src/app/Services/codeql-service';
import { FilesService } from 'src/app/Services/fileService';
import { SocketService } from 'src/app/Services/socket-service.service';
import { CVEUtilService } from 'src/app/attack-surface/cwe-util.service';
import { EditorService } from 'src/app/shared-components/editor-service';
import {ConfirmationDialogComponent} from './dialogs/confirmation-dialog.component';
import { MatDialog } from '@angular/material/dialog';
import { ChatGptService } from 'src/app/Services/chatgpt.service';

declare var $: any;
@Component({
    selector: 'app-page-header',
    templateUrl: './page-header.component.html',
})
export class PageHeaderComponent implements OnInit {
    isLoading:boolean=false;
    calculatedCost: number | null = null;
    constructor(
        public utilService: CVEUtilService,
        private socketService:SocketService,
        private fileService: FilesService,
        private codeQlService:CodeQlService,
        private chatGptService:ChatGptService,
        private editorService:EditorService,
        public dialog: MatDialog

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

        // this.openAgreementDialog();
    }

    uploadProject() {
        var data: FormData = new FormData();
        data.append('file', this.utilService.SelectedProject, this.utilService.ProjectName);
        this.isLoading=true;

        this.fileService.uploadFile(data).subscribe((response)=>{
            this.utilService.directoryFilesTree=response;
            this.utilService.isUploaded=true;
            this.isLoading=false;
        })
    }


    async runCodeQl(){
        this.socketService.clearOutput();
        await this.socketService.socketConnect();
        this.isLoading=true;
        
        $('#terminal').toggleClass('d-none');

        this.codeQlService.runCodeQl({project:this.utilService.ProjectName}).subscribe((response)=>{
            this.editorService.rulesTree=response.rulesTree;
            this.editorService.locationsTree=response.locationsTree;
            this.isLoading=false;
            this.socketService.socketDisconnect();
            // $('#terminal').toggleClass('d-none');

        });
    }

    async usePrevious(){
        this.socketService.clearOutput();
        await this.socketService.socketConnect();
        this.isLoading=true;

            this.codeQlService.getSarifResult(this.utilService.ProjectName).subscribe((response)=>{
                try{
                    this.editorService.rulesTree=response.rulesTree;
                    console.log(response.rulesTree);
                    this.editorService.locationsTree=response.locationsTree;
                    this.isLoading=false;
                    this.socketService.socketDisconnect();
                }catch(e){
                    this.isLoading=false;
                    this.socketService.socketDisconnect();
                }
        });
    }   

    openAgreementDialog(): void {
        const dialogRef = this.dialog.open(ConfirmationDialogComponent, {
          height: '235px',
          width: '500px',
          position: {
            top: '100px'
          },
          data: {
            onConfirm: () => this.runCodeQl(),
            isLoading: true,
            cost : this.calculatedCost
          }
        });
        if (this.calculatedCost === null){
          this.getCost().then(cost => {
            dialogRef.componentInstance.updateData({ isLoading: false, cost });
          });
        } else {
          dialogRef.componentInstance.data.isLoading = false;
          dialogRef.componentInstance.data.cost = this.calculatedCost;
      }
      
        
      
        // dialogRef.afterClosed().subscribe(result => {
        //   if (result) {
        //     this.utilService.export();
        //   }
        // });
      }
      
      async getCost(): Promise<number> {
        this.socketService.clearOutput();
        await this.socketService.socketConnect();
    
        return new Promise((resolve, reject) => {
          this.chatGptService.getCostEstimate(this.utilService.ProjectName).subscribe(
            // this.chatGptService.getCostEstimate("CWEToyDataset").subscribe(
            (response) => {
              resolve(response.totalCost);
              this.socketService.socketDisconnect();
            },
            (error) => {
              reject(error);
              this.socketService.socketDisconnect();
            }
          );
        });
      }
      
}
