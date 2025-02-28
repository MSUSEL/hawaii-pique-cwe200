import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { CodeQlService } from './codeql-service'; // Assuming this is where your CodeQl API calls are

@Injectable({
  providedIn: 'root',
})
export class DataFlowService {
  private dataFlowChange = new BehaviorSubject<any>(null);
  private dataFlowLabelMap = new Map<number, Map<number, string>>();

  // Used to labeling the data flow as "True Positive" or "False Positive"
  // Initialize the label map for a tree
  addLabel(tree: any[]): void {
    console.log(tree)
    
    if (!tree || tree.length === 0 || tree[0].length === 0) {
      console.warn('Empty tree provided to addLabel');
      return;
    }

    // Fix the typo: vulnerbilityIndex -> vulnerabilityIndex
    // Also convert it to string if it's not already, for consistent keys
    const vulIndex = Number(tree[0][0].vulnerabilityIndex) // This is the object containing vulnerabilityIndex
    
    // Check for both possible spellings of the property
    console.log('VulIndex:', vulIndex, 'Type:', typeof vulIndex);
    
    // Initialize maps for each flow in the vulnerability if not exists
    if (!this.dataFlowLabelMap.has(vulIndex)) {
      this.dataFlowLabelMap.set(vulIndex, new Map<number, string>());
      
      // Initialize entries for each flow index in this vulnerability
      for (const flow of tree) {
        if (flow.length > 0) {
          const flowIndex = flow[0].flowIndex;
          if (flowIndex !== undefined && !this.dataFlowLabelMap.get(vulIndex)!.has(flowIndex)) {
            // Initialize with empty label
            this.dataFlowLabelMap.get(vulIndex)!.set(flowIndex, '');
          }
        }
      }
    }

    // Log the current state of the label map
    console.log('Current label map:', this.dataFlowLabelMap);
  }

  // Get the total count of labeled flows across all vulnerabilities
getLabeledFlowCount(): number {
  let count = 0;
  this.dataFlowLabelMap.forEach(flowMap => {
    count += flowMap.size;
  });
  return count;
}

// Update the updateLabel method to ensure it's correctly storing labels
updateLabel(vulIndex: number, flowIndex: number, label: string): void {
  console.log(`Updating label for vulIndex: ${vulIndex}, flowIndex: ${flowIndex}, label: ${label}`);
  
  // Make sure vulIndex is a number and convert to string for consistent map keys
  const vulIndexKey = vulIndex;
  
  if (!this.dataFlowLabelMap.has(vulIndexKey)) {
    console.log(`Creating new map for vulIndex: ${vulIndexKey}`);
    this.dataFlowLabelMap.set(vulIndexKey, new Map<number, string>());
  }
  
  this.dataFlowLabelMap.get(vulIndexKey)!.set(flowIndex, label);
  
  // Log the updated map state for debugging
  console.log('Updated label map:', this.getDebugLabelMap());
}

// Add this helper method to visualize the map
getDebugLabelMap(): any {
  const debugMap: {[vulIndex: number]: {[flowIndex: number]: string}} = {};
  
  this.dataFlowLabelMap.forEach((flowMap, vulIndex) => {
    debugMap[vulIndex] = {};
    flowMap.forEach((label, flowIndex) => {
      debugMap[vulIndex][flowIndex] = label;
    });
  });
  
  return debugMap;
}
  getLabel(vulIndex: number, flowIndex: number): string | undefined {
    return this.dataFlowLabelMap.get(vulIndex)?.get(flowIndex);
  }

  // Get the complete map of labels for submission
getCompleteDataFlowLabelMap(): Map<number, Map<number, string>> {
  return this.dataFlowLabelMap;
}

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
          let tree = this.convertDataToTree(data, node.index);
          this.addLabel(tree)
          console.log(tree)
          this.dataFlowChange.next(tree);  // Update the data flow with the real data
          
        },
        error: (error) => {
          console.error('Error fetching data flow:', error);
        }
      });
  }

  // Helper function to format the real data into the tree structure expected by the component
  private convertDataToTree(data: any[], vulIndex): any[] {
    // console.log('Converting Data:', data);

    // Group flows separately into their own arrays
    return data.map((flow, flowIndex) => {
      // console.log(`Processing Flow ${flowIndex}`);

      // Map each node in the current flow to its respective FlowNode structure
      const flowNodes = Object.keys(flow).map(key => {
        const node = flow[key];
        // console.log(`Flow ${flowIndex}, Node ${key}:`, node);

        return {
          vulnerabilityIndex: vulIndex,
          flowIndex: flowIndex,
          label: node.label || '',
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

  submitFlowLabels(labelData: any) {
    console.log('Sending all labels to backend:', labelData);
    // This sends the entire map of labeled flows to the backend
    this.codeQlService.submitFlowLabels(labelData).subscribe(
      response => console.log('Response:', response),
      error => console.error('Error:', error)
    );;
  }
}
