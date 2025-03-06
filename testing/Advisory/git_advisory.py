import requests
import collections
import os
import re

# Load GitHub Token from Environment Variable
GITHUB_TOKEN = ''

# GitHub API Endpoint
GITHUB_API_URL = "https://api.github.com/advisories"

# Headers for Authentication
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}"
}

# Parameters for Filtering (CWE-200, Java/Maven)
params = {
    "cwe_id": "CWE-200",
    "ecosystem": "maven",
    "per_page": 100  # Fetch the maximum number of results per page
}

# Function to Fetch Advisories with Proper Pagination
def fetch_advisories():
    advisories = []
    url = GITHUB_API_URL  # Start with the base URL

    while url:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=HEADERS, params=params if url == GITHUB_API_URL else None)

        if response.status_code != 200:
            print(f"❌ Error fetching data: {response.status_code}")
            break

        data = response.json()
        if not data:
            break

        advisories.extend(data)

        # Check for pagination in headers
        link_header = response.headers.get("Link", "")
        next_url = None

        if link_header:
            links = link_header.split(", ")
            for link in links:
                if 'rel="next"' in link:
                    next_url = link.split(";")[0].strip("<>")

        url = next_url  # Continue fetching the next page, or stop if None

    return advisories

# Function to Extract Repository Name from Fix Commit URL
def extract_repo_from_commit(commit_urls):
    for url in commit_urls:
        match = re.search(r"https://github\.com/([^/]+/[^/]+)/commit/", url)
        if match:
            return match.group(1)  # Return first match
    return None  # No valid repo found in commit URLs

# Fetch advisories with proper pagination
all_advisories = fetch_advisories()

# Dictionary to group advisories by repository while ensuring unique CVEs
projects = collections.defaultdict(dict)

for advisory in all_advisories:
    repo_name = "Unknown"  # Default if no valid repo is found
    fix_commits = None

    # 1️⃣ Try to get repo_name from `source_code_location`
    repo_url = advisory.get("source_code_location", "")

    if isinstance(repo_url, str) and repo_url.startswith("https://github.com/"):
        repo_name = "/".join(repo_url.split("/")[-2:])  

    else:
        # 2️⃣ Try to get repo_name from `package_name`
        package_name = advisory.get("package_name", "").strip()
        if isinstance(package_name, str) and "/" in package_name:
            repo_name = package_name  # Format is usually "groupId/artifactId"

        else:
            # 3️⃣ Try to get repo_name from `fix commit URLs`
            fix_commits = [ref for ref in advisory.get("references", []) if isinstance(ref, str) and "/commit/" in ref]
            extracted_repo = extract_repo_from_commit(fix_commits)

            if isinstance(extracted_repo, str):
                repo_name = extracted_repo
            else:
                # 4️⃣ Try to get from `identifiers`
                identifiers = advisory.get("identifiers", [])
                if identifiers and isinstance(identifiers[0], str):
                    repo_name = identifiers[0]

    # Ensure repo_name is always a string
    repo_name = str(repo_name)  

    # Only store advisories with unique CVEs per repository
    cve_id = advisory.get("cve_id", "Unknown_CVE")  # Avoid missing keys
    if isinstance(cve_id, str) and cve_id not in projects[repo_name]:  
        projects[repo_name][cve_id] = {
            "ID": cve_id,
            "Summary": advisory.get("summary", "No summary available"),
            "Severity": advisory.get("severity", "Unknown"),
            "Published": advisory.get("published_at", "Unknown date"),
            "Fix Commits": fix_commits if fix_commits else ["No commit found"],
            "Advisory URL": advisory.get("html_url", "No URL available")
        }

# Save full advisory details to a text file
with open(os.path.join("testing", "Advisory", "advisories.txt"), "w", encoding="utf-8") as file:
    for project, advisories in projects.items():
        file.write(f"\n🔹 Repository: {project} ({len(advisories)} unique advisories)\n")
        file.write("-" * 50 + "\n")
        for adv in advisories.values():  # Iterate through unique advisories
            file.write(f"  - 🆔 {adv['ID']} | {adv['Summary']} | Severity: {adv['Severity']}\n")
            file.write(f"    📅 Published: {adv['Published']}\n")
            file.write(f"    🔗 Advisory: {adv['Advisory URL']}\n")
            file.write(f"    🔄 Fix Commits: {', '.join(adv['Fix Commits'])}\n\n")

num = 150
with open(os.path.join("testing", "Advisory", "top_advisories.txt"), "w", encoding="utf-8") as file:
# Display Top num Projects with the Most Unique Advisories
    file.write(f"\n📊 Top {num} Projects with the Most Unique Advisories 📊\n")
    sorted_projects = sorted(projects.items(), key=lambda x: len(x[1]), reverse=True)[:num]

    for rank, (project, unique_advisories) in enumerate(sorted_projects, start=1):
        file.write(f"{rank}. {project} - {len(unique_advisories)} unique advisories\n")

    file.write("\n✅ Full advisory details saved to 'advisories.txt'\n")
