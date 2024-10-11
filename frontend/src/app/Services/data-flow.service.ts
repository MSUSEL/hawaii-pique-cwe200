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
  }

  // Helper function to format the real data into the tree structure expected by the component
  private convertDataToTree(data: any[]): any[] {
    console.log('Converting Data:', data);

    // Group flows separately into their own arrays
    return data.map((flow, flowIndex) => {
      console.log(`Processing Flow ${flowIndex}`);

      // Map each node in the current flow to its respective FlowNode structure
      const flowNodes = Object.keys(flow).map(key => {
        const node = flow[key];
        console.log(`Flow ${flowIndex}, Node ${key}:`, node);

        return {
          message: node.message || '',
          uri: node.uri || '',
          startLine: node.startLine || 0,
          startColumn: node.startColumn || 0,
          endColumn: node.endColumn || 0,
          endLine: node.endLine || node.startLine || 0,
          type: node.type || '',
        };
      });

      // Return each flow as an array of nodes
      return flowNodes;
    });
  }
}
