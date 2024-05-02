import requests
import os
import json
import backoff
import time
import csv

# This script queries the NVD API for vulnerabilities associated with CWE-200 in java projects and writes the results to a CSV file.

def main():
    cwes = get_cwes('backend/Files/CWEToyDataset/src/main/java/com/mycompany/app/')
    results = []
    for cwe in cwes:
        results.append(query_nvd(cwe))
    write_to_csv(results, "./testing/NVD/vulnerabilities.csv")
        
def get_cwes(dir):
    cwes = []
    for cwe in os.listdir(dir):
        cwes.append(cwe)
    return cwes


def query_nvd(cwe):
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    query = f"{base_url}?cweId={cwe}&keywordSearch=Java"
    retries = 5  # Maximum number of retries
    backoff_factor = 5  # Initial backoff duration in seconds
    

    for attempt in range(retries):
        response = requests.get(query)

        if response.status_code == 200:
            # Parse the JSON data from the response
            data = response.json()
            if len(data['vulnerabilities']) > 0:
                # Assume parse_nvd_data function processes data
                # print(f"Found {len(data['vulnerabilities'])} vulnerabilities for {cwe}.")
                return parse_nvd_data(data['vulnerabilities'], cwe) 
            else:
                # print(f"No vulnerabilities found for {cwe} that meet the criteria.")
                return []
        
        elif response.status_code == 403:
                time.sleep(backoff_factor)
                backoff_factor *= 2
                # print(f"Failed to retrieve data for {cwe}: HTTP {response.status_code} - Retrying in {backoff_factor} seconds.")
    return []

def parse_nvd_data(data, cweID):
    query_results = []
    for cve in data:
        product_names = []

        cwe = cve['cve']['weaknesses'][0]['description'][0]['value']
        cve_id = cve['cve']['id']
        descrption = cve['cve']['descriptions'][0]['value']
        products = cve['cve']['configurations'][0]['nodes']
        
        for pro in products:
            pro = extract_product_name(pro['cpeMatch'][0]['criteria'])
            product_names.append(pro)

        query_results.append({"cwe": cwe,
                      "cve": cve_id, 
                      "description": descrption, 
                      "products": product_names})
        
        if cwe == cweID:
            print(f"{cwe} mapped to {cve_id} found in {product_names[0]}" )
    return query_results
        


def extract_product_name(cpe_uri):
    parts = cpe_uri.split(":")
    if len(parts) > 4:
        return f"{parts[4]} version {parts[5]} "  # Return the product name part
    return "Unknown"  # Return Unknown if the CPE URI doesn't contain enough parts


def write_to_csv(data, filename):
    # Extract relevant fields from the JSON data
    rows = []
    for results in data:
        for item in results:
            cwe = item['cwe']
            cve = item['cve']
            description = item['description']
            products = item['products']

            # Construct NVD link
            nvd_link = f"https://nvd.nist.gov/vuln/detail/{cve}"

            rows.append([cwe, cve, products, description, nvd_link])

    # Write data to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['CWE', 'CVE ID', 'Products', 'Description', 'NVD Link'])
        writer.writerows(rows)
if "__main__" == __name__:
    main()

