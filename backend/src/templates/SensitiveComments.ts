// Defines the structure of the sensitive comments data that is used in the .yml files.
export const SensitiveComments:string=`
extensions:  
  - addsTo:
      pack: custom-codeql-queries
      extensible: sensitiveComments
    data:
    - ["fileName", "senstiveComment"]
**********
`