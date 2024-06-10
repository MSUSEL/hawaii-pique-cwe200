import { FlatTreeControl } from "@angular/cdk/tree";
import {
    Component,
    Directive,
    Input,
    OnChanges,
    OnInit,
    SimpleChanges,
    Output,
    EventEmitter,
} from "@angular/core";
import {
    MatTreeFlatDataSource,
    MatTreeFlattener,
} from "@angular/material/tree";
import { EditorService } from "../editor-service";


export interface CodeQlNode {
    files: CodeQlNode[];
    name: string;
    type:string;
    fullPath:string;
    message:string;
    region:any;
    location:any;
}
export class RuleFlatNode {
    name: string;
    level: number;
    type:string;
    fullPath:string;
    message:string;
    expandable: boolean;
    size:number=0;
    location:string;
}

@Directive()
export class RuleTree {
    constructor(protected editorService:EditorService) {}

    hasChild = (_: number, node: RuleFlatNode) => node.expandable;
    getLevel = (node: RuleFlatNode) => node.level;

    protected _transformer = (node: CodeQlNode, level: number) => {
        return {
            expandable: !!node.files && node.files.length > 0,
            name: node.name,
            level: level,
            type: node.type,
            message:node.message,
            fullPath:node.fullPath?node.fullPath:'',
            size:node.files?node.files.length:0,
            region:node.region,
            location:node.location
        };
    };

    treeControl = new FlatTreeControl<RuleFlatNode>(
        (node) => node.level,
        (node) => node.expandable
    );

    treeFlattener = new MatTreeFlattener(
        this._transformer,
        (node) => node.level,
        (node) => node.expandable,
        (node) => node.files
    );

    dataSource = new MatTreeFlatDataSource(
        this.treeControl,
        this.treeFlattener
    );
}

@Component({
    selector: "app-rule-tree",
    templateUrl: "./rule-tree.html",
    styleUrls: ["./tree.scss"],
})
export class RuleTreeComponent extends RuleTree implements OnInit {
    SelectedNode:any=null;
    
    @Input() treeData: any[] = [];
    @Output() viewChange: EventEmitter<any> = new EventEmitter<any>();
    ngOnInit() {
        this.dataSource.data = this.treeData;
    }
    ngOnChanges(changes: SimpleChanges) {
        if (changes['treeData'] && changes['treeData'].currentValue) {
            this.dataSource.data = this.treeData;
            //this.treeControl.expandAll();
        }
    }

    findFile(node:any){
        this.editorService.findFile(node);
    }
}


@Component({
    selector: "app-location-tree",
    templateUrl: "./locations-tree.html",
    styleUrls: ["./tree.scss"],
})
export class LocationsTreeComponent extends RuleTree implements OnInit {
    SelectedNode:any=null;
    @Input() treeData: any[] = [];
    ngOnInit() {
        this.dataSource.data = this.treeData;
    }
    ngOnChanges(changes: SimpleChanges) {
        if (changes['treeData'] && changes['treeData'].currentValue) {
            this.dataSource.data = this.treeData;
        }
    }

    findFile(node:any){
        this.editorService.findFile(node);
    }
}


