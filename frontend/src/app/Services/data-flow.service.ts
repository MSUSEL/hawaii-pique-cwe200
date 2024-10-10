import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class DataFlowService {
  private dataFlowChange = new BehaviorSubject<any>(null);

  // Expose the observable
  public dataFlowChangeObservable = this.dataFlowChange.asObservable();

  // Method to update the data flow
  findFlow(node: any) {
    console.log(JSON.stringify(node, null, 2));  // Pretty print the node object
    const pathComponents: string[] = node.fullPath.split(/[/\\]+/);
    const project = pathComponents[1];
    const vulnerabilityId = node.fullPath;
    const region = node.region;

    // Commenting out the real API call for now and using dummy data for testing
    /*
    this.codeQlService.getDataFlowTree(vulnerabilityId, project, region)  // Call the CodeQlService method
      .pipe(
        map(response => {
          console.log('Vulnerability Tree Response:', response);
          return response;  // Process the response as needed
        })
      )
      .subscribe({
        next: (data) => {
          this.dataFlowChange.next(data);  // Update the data flow with the API response
        },
        error: (error) => {
          console.error('Error fetching data flow:', error);
        }
      });
    */

    // Dummy data for testing
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
  }

  // Helper function to format the dummy data into the tree structure expected by the component
  private convertDummyDataToTree(data: any) {
    return Object.keys(data).map(key => ({
      message: `${data[key].message}`,
      uri: data[key].uri,
      Line: data[key].Line,
      Column: data[key].Column
    }));
  }
}
