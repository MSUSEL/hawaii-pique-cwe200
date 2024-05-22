import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HttpClientModule , HTTP_INTERCEPTORS} from '@angular/common/http';
import {
    CommonModule,
    HashLocationStrategy,
    LocationStrategy,
} from '@angular/common';
import { AppRoutingModule } from './app-routing.module';
import { AttackSurfaceModule } from './attack-surface/cwe.module';

import { AppComponent } from './app.component';
import { TopHeaderComponent } from './home/top-header/top-header.component';
import { FooterComponent } from './home/footer/footer.component';
import { AppInterceptorService } from './Services/InterCeptors';
import { SharedComponentsModule } from './shared-components/shared-components.module';


@NgModule({
    declarations: [AppComponent, TopHeaderComponent, FooterComponent],
    imports: [
        CommonModule,
        BrowserModule,
        ReactiveFormsModule,
        FormsModule,
        BrowserAnimationsModule,
        HttpClientModule,
        AppRoutingModule,
        SharedComponentsModule,
        AttackSurfaceModule,
        
    ],
    providers: [
        { provide: LocationStrategy, useClass: HashLocationStrategy },
        {
            provide: HTTP_INTERCEPTORS,
            useClass: AppInterceptorService,
            multi: true,
        },
    ],
    bootstrap: [AppComponent],
})
export class AppModule {}
