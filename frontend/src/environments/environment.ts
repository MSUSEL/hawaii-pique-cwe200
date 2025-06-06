// This file can be replaced during build by using the `fileReplacements` array.
// `ng build` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.


// This defines the what the backend API server URL uses.
// The frontend defaults to http://localhost:4200
export const environment = {
    production: false,
    // apiUrl: 'http://www.sain.info:5400',
    // apiUrl: 'http://129.21.128.6:5400',
    apiUrl: 'http://localhost:5400',

};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/plugins/zone-error';  // Included with Angular CLI.
