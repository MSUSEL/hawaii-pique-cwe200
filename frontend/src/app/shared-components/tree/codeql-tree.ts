import { FlatTreeControl } from "@angular/cdk/tree";
import {
    Component,
    Directive,
    Input,
    OnChanges,
    OnInit,
    SimpleChanges,
    Output,
    EventEmitter
} from "@angular/core";
import {
    MatTreeFlatDataSource,
    MatTreeFlattener
} from "@angular/material/tree";
import { EditorService } from "../editor-service";
import { DataFlowService } from '../../Services/data-flow.service'; // Import the DataFlowService

export interface CodeQlNode {
    files: CodeQlNode[];
    name: string;
    type: string;
    fullPath: string;
    message: string;
    region: any;
    location: any;
    index: string;
}

export class RuleFlatNode {
    name: string;
    level: number;
    type: string;
    fullPath: string;
    message: string;
    expandable: boolean;
    size: number = 0;
    location: string;
    index: string;
}

@Directive()
export class RuleTree {
    treeControl: FlatTreeControl<RuleFlatNode>;
    treeFlattener: MatTreeFlattener<CodeQlNode, RuleFlatNode>;
    dataSource: MatTreeFlatDataSource<CodeQlNode, RuleFlatNode>;

    constructor(protected editorService: EditorService) {
        this.treeFlattener = new MatTreeFlattener(
            this._transformer,
            (node) => node.level,
            (node) => node.expandable,
            (node) => node.files
        );

        this.treeControl = new FlatTreeControl<RuleFlatNode>(
            (node) => node.level,
            (node) => node.expandable
        );

        this.dataSource = new MatTreeFlatDataSource(
            this.treeControl,
            this.treeFlattener
        );
    }

    hasChild = (_: number, node: RuleFlatNode) => node.expandable;
    getLevel = (node: RuleFlatNode) => node.level;

    protected _transformer = (node: CodeQlNode, level: number) => {
        return {
            expandable: !!node.files && node.files.length > 0,
            name: node.name,
            level: level,
            type: node.type,
            message: node.message,
            fullPath: node.fullPath ? node.fullPath : '',
            size: node.files ? node.files.length : 0,
            region: node.region,
            location: node.location,
            index: node.index
        };
    };
}

@Component({
    selector: "app-rule-tree",
    templateUrl: "./rule-tree.html",
    styleUrls: ["./tree.scss"],
})
export class RuleTreeComponent extends RuleTree implements OnInit, OnChanges {
    @Input() treeData: any[] = [];
    @Output() nodeSelected = new EventEmitter<any>();

    constructor(
        protected override editorService: EditorService, // Add 'override' keyword
        private dataFlowService: DataFlowService // Inject the DataFlowService
    ) {
        super(editorService);
    }

    ngOnInit() {
        this.dataSource.data = this.treeData;
    }

    ngOnChanges(changes: SimpleChanges) {
        if (changes['treeData'] && changes['treeData'].currentValue) {
            this.dataSource.data = this.treeData;
        }
    }

    findFile(node: any) {
        // Find the file in the editor
        this.editorService.findFile(node);
        this.dataFlowService.findFlow(node)
        console.log('Codeql Tree:', node);


        // Emit the selected node
        this.nodeSelected.emit(node);
    }
}

@Component({
    selector: "app-location-tree",
    templateUrl: "./locations-tree.html",
    styleUrls: ["./tree.scss"],
})
export class LocationsTreeComponent extends RuleTree implements OnInit, OnChanges {
    @Input() treeData: any[] = [];
    @Output() nodeSelected = new EventEmitter<any>();

    constructor(
        protected override editorService: EditorService, // Add 'override' keyword
        private dataFlowService: DataFlowService // Inject the DataFlowService
    ) {
        super(editorService);
    }

    ngOnInit() {
        this.dataSource.data = this.treeData;
    }

    ngOnChanges(changes: SimpleChanges) {
        if (changes['treeData'] && changes['treeData'].currentValue) {
            this.dataSource.data = this.treeData;
        }
    }

    findFile(node: any) {
        // Find the file in the editor
        this.editorService.findFile(node);
        console.log('Node clicked in locations:', node);

        // Emit the selected node
        this.nodeSelected.emit(node);
    }
}
