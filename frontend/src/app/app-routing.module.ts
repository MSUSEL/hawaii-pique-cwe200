import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AttackSurfaceComponent } from './attack-surface/main.component';


const routes: Routes = [
    { path: '', redirectTo: '/cwe', pathMatch: 'full' },
    {
        path: 'cwe',
        component: AttackSurfaceComponent,
    },
];

@NgModule({
    imports: [RouterModule.forRoot(routes,{
        useHash: true 
    })],
    exports: [RouterModule],
})
export class AppRoutingModule {}
