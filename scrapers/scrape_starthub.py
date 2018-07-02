#scraper for starthub

import urllib.request as url
from bs4 import BeautifulSoup
import csv
from tqdm import trange


outfile= "../data/starthub.csv"

#number of pages from the site
NUM_PAGES=281

company_list = []

for i in trange(1, NUM_PAGES):
    page_url = "https://www.starthub.org/startups?page={}".format(i)

    content = url.urlopen(page_url)

    soup = BeautifulSoup(content, "html.parser")

    #get list of companies
    companies = soup.findAll("article", attrs={"class": "node node-company node-teaser slat slat_right search-result"})


    #converting to dict for easy CSV parsing
    for company in companies:
        name = company.find("a", attrs={"class":"dark"}).text.strip()
        description = company.find("p").text.strip().replace("\n", " ").lower()

        if len(description) > 0:
            company_list.append({"name":name, "description":description})

#saving files
with open(outfile, "w") as f:
    writer = csv.DictWriter(f, ['name', 'description'])
    writer.writeheader()
    writer.writerows(company_list)
