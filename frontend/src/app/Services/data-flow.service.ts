import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { CodeQlService } from './codeql-service'; // Assuming this is where your CodeQl API calls are

@Injectable({
  providedIn: 'root',
})
export class DataFlowService {
  private dataFlowChange = new BehaviorSubject<any>(null);

  // Expose the observable
  public dataFlowChangeObservable = this.dataFlowChange.asObservable();

  constructor(private codeQlService: CodeQlService) {}

  // Method to update the data flow
  findFlow(node: any) {
    console.log(JSON.stringify(node, null, 2));  // Pretty print the node object
    const pathComponents: string[] = node.fullPath.split(/[/\\]+/);
    const project = pathComponents[1]; // Extract project from path
    const vulnerabilityId = node.fullPath; // Use the full path as the vulnerability ID
    const region = node.region; // Get the region from the node

    // Call the CodeQlService to get the data flow tree
    this.codeQlService.getDataFlowTree(vulnerabilityId, project, node.index)
      .subscribe({
        next: (data) => {
          console.log('Vulnerability Tree Response:', data);
          this.dataFlowChange.next(this.convertDataToTree(data));  // Update the data flow with the real data
        },
        error: (error) => {
          console.error('Error fetching data flow:', error);
        }
      });

    // Dummy data for testing (commented out)
    /*
    const dummyData = {
      "0": {
        "message": "doFinal(...) : byte[]",
        "uri": "CWEToyDataset/src/main/java/com/mycompany/app/CWE-208/AttackAgainstSigniture/BAD_AAS_unsafeCheckCiphertext.java",
        "Line": 15,
        "Column": 26
      },
      "1": {
        "message": "tag",
        "uri": "CWEToyDataset/src/main/java/com/mycompany/app/CWE-208/AttackAgainstSigniture/BAD_AAS_unsafeCheckCiphertext.java",
        "Line": 17,
        "Column": 49
      }
    };

    // Simulating an API response with dummy data
    this.dataFlowChange.next(this.convertDummyDataToTree(dummyData));  // Update the data flow with dummy data
    */
  }

  // Helper function to format the real data into the tree structure expected by the component
  private convertDataToTree(data: any) {
    return Object.keys(data).map(key => ({
      message: `${data[key].message}`,
      uri: data[key].uri,
      Line: data[key].Line,
      Column: data[key].Column
    }));
  }

  // Helper function for dummy data (commented out)
  /*
  private convertDummyDataToTree(data: any) {
    return Object.keys(data).map(key => ({
      message: `${data[key].message}`,
      uri: data[key].uri,
      Line: data[key].Line,
      Column: data[key].Column
    }));
  }
  */
}
