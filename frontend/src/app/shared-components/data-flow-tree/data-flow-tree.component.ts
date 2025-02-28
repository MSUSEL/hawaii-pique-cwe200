import { Component, Input, OnInit } from '@angular/core';
import { DataFlowService } from 'src/app/Services/data-flow.service';
import { EditorService } from 'src/app/shared-components/editor-service';  // Import the EditorService
import { ItemFlatNode } from 'src/app/shared-components/tree/tree.component'; // Import ItemFlatNode

export interface FlowNode {
  message: string;  // The message to display
  uri: string;      // The file URI (not displayed, but kept for navigation)
  startLine: number; // The starting line number for the highlight
  startColumn: number; // The starting column number for the highlight
  endLine: number; // The ending line number for the highlight
  endColumn: number; // The ending column number for the highlight
  isExpanded?: boolean; // To track the expansion state of the node
  type: string; // The type of the node
  vulnerabilityIndex?: string;
  flowIndex?: number;
}

@Component({
  selector: 'app-data-flow-tree',
  templateUrl: './data-flow-tree.component.html',
  styleUrls: ['./data-flow-tree.component.scss']
})
export class DataFlowTreeComponent implements OnInit {
  @Input() treeData: FlowNode[][] = [];  // Now an array of arrays for separate flows
  activeFlowIndex: number = -1;  // Track the last clicked flow tab
  activeNodeIndex: number = -1;  // Track the last clicked node within a flow
  
  // Store current vulnerability index and project
  currentVulIndex: number = -1;
  currentProject: string = '';

  constructor(
    public dataFlowService: DataFlowService,
    private editorService: EditorService  // Inject the EditorService
  ) {}

  ngOnInit(): void {
    // Subscribe to the data flow change observable and update the tree
    this.dataFlowService.dataFlowChangeObservable.subscribe((data) => {
      if (data) {
        this.treeData = data.map(flow => flow.map(node => ({ ...node, isExpanded: false })));  // Initialize all nodes as collapsed
        this.activeFlowIndex = -1; // Reset the active flow when a new vulnerability is clicked
        this.activeNodeIndex = -1; // Reset the active node when a new vulnerability is clicked
        
        // Extract vulnerability index and project from the first node of first flow
        if (this.treeData.length > 0 && this.treeData[0].length > 0) {

          console.log('Tree data at this point ', this.treeData)

          const firstNode = this.treeData[0][0];
          // Make sure to explicitly convert to number to avoid type issues
          this.currentVulIndex = Number(this.treeData[0][0].vulnerabilityIndex)
          
          const uri = firstNode.uri || '';
          this.currentProject = uri.split(/[/\\]+/)[0];
          
          console.log(`Current vulnerability: ${this.currentVulIndex}, Project: ${this.currentProject}`);
        }
      }
    });
  }
  

  // Toggle flow tab expansion and allow collapsing
  toggleFlow(flowIndex: number): void {
    if (this.activeFlowIndex === flowIndex) {
      this.activeFlowIndex = -1;  // Collapse if already expanded
    } else {
      this.activeFlowIndex = flowIndex;  // Expand the selected flow
      this.activeNodeIndex = -1;  // Reset node highlight when switching flows
    }
  }

  // Toggle node expansion and trigger highlight in editor
  onNodeClick(flowIndex: number, nodeIndex: number): void {
    // Collapse previously expanded nodes if needed
    if (this.activeNodeIndex !== -1 && this.activeFlowIndex === flowIndex) {
      this.treeData[flowIndex][this.activeNodeIndex].isExpanded = false;
    }

    const node = this.treeData[flowIndex][nodeIndex];
    node.isExpanded = !node.isExpanded;  // Toggle expansion

    // Set active node index to apply highlighting
    this.activeNodeIndex = nodeIndex;

    const fullPath = this.correctPath(node.uri);

    const fileNode: ItemFlatNode = {
      name: fullPath.split('/').pop(),
      fullPath: fullPath,
      level: 0,
      type: node.type,
      expandable: false,
      code: '',
      region: {
        startLine: node.startLine,
        startColumn: node.startColumn,
        endLine: node.endLine,
        endColumn: node.endColumn
      }
    };

    // Find and highlight the file in the editor
    this.editorService.findFile(fileNode, () => {
      // The actual highlight is handled in the CodeEditorComponent
    });
  }

  // Helper to correct file path format
  correctPath(uri: string): string {
    const pathComponents = uri.split(/[/\\]+/);
    const projectName = pathComponents[0];
    let fullPath = uri;
    if (!fullPath.startsWith('Files/')) {
      fullPath = `Files/${projectName}/${fullPath}`;
    }
    return fullPath;
  }
  
 // Label a flow as vulnerable (Yes) or not (No)
 labelFlow(flowIndex: number, isVulnerable: boolean): void {
  if (this.currentVulIndex === -1) {
    console.warn('Cannot label flow: currentVulIndex not set');
    return;
  }
  
  const label = isVulnerable ? 'Yes' : 'No';
  console.log(`Labeling flow ${flowIndex} as ${label} for vulnerability ${this.currentVulIndex}`);
  this.dataFlowService.updateLabel(this.currentVulIndex, flowIndex, label);
}

// Check if a flow has been labeled
isFlowLabeled(flowIndex: number): boolean {
  if (this.currentVulIndex === -1) return false;
  const label = this.dataFlowService.getLabel(this.currentVulIndex, flowIndex);
  return label !== undefined && label !== '';
}

// Get the label for a flow
getFlowLabel(flowIndex: number): string | undefined {
  if (this.currentVulIndex === -1) return undefined;
  return this.dataFlowService.getLabel(this.currentVulIndex, flowIndex);
}

// Check if a flow is labeled as vulnerable (Yes)
isVulnerable(flowIndex: number): boolean {
  return this.getFlowLabel(flowIndex) === 'Yes';
}

// Check if a flow is labeled as not vulnerable (No)
isNotVulnerable(flowIndex: number): boolean {
  return this.getFlowLabel(flowIndex) === 'No';
}

getLabeledFlowCount(): number {
  return this.dataFlowService.getLabeledFlowCount();
}

// Remove the areAllFlowsLabeled method since we don't need it anymore

// Modify the submitAllLabels method to handle the new labels
submitAllLabels(): void {
  // Get the complete map from the service
  const completeMap = this.dataFlowService.getCompleteDataFlowLabelMap();
  
  if (!completeMap || completeMap.size === 0) {
    console.warn('No labels to submit');
    return;
  }
  
  // Prepare the data for submission with all vulnerabilities and their flows
  const allLabelsData = {
    project: this.currentProject,
    vulnerabilities: Array.from(completeMap.entries()).map(([vulIndex, flowLabels]) => {
      return {
        vulnerabilityId: vulIndex.toString(),
        flows: Array.from(flowLabels.entries()).map(([flowIndex, label]) => {
          return {
            flowIndex: flowIndex,
            isVulnerable: label === 'Yes',
            label: label
          };
        })
      };
    })
  };

  console.log('Submitting all flow labels:', allLabelsData);
  this.dataFlowService.submitFlowLabels(allLabelsData);
}
}