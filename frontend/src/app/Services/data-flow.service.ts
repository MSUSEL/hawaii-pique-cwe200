import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { CodeQlService } from './codeql-service'; // Import the CodeQlService
import { map } from 'rxjs/operators';


@Injectable({
  providedIn: 'root',
})
export class DataFlowService {
  private dataFlowChange = new BehaviorSubject<any>(null);

  // Expose the observable
  public dataFlowChangeObservable: Observable<any>;

  constructor(private codeQlService: CodeQlService) {  // Inject CodeQlService
    this.dataFlowChangeObservable = this.dataFlowChange.asObservable();
  }

  // Method to update the data flow, by calling CodeQlService
  findFlow(node: any) {
    console.log('DataFlowService: calling getVulnerabilityTree' + node.fullPath + ' ' + node.name);
    
    // this.codeQlService.getVulnerabilityTree(vulnerabilityId, project)
    //   .pipe(
    //     map(response => {
    //       console.log('Vulnerability Tree Response:', response);
    //       return response;  // Process the response as needed
    //     })
    //   )
    //   .subscribe({
    //     next: (data) => {
    //       this.dataFlowChange.next(data);  // Update the data flow with the API response
    //     },
    //     error: (error) => {
    //       console.error('Error fetching data flow:', error);
    //     }
    //   });
  }
}
