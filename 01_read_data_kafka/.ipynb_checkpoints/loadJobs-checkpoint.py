

import psycopg2
import time
import requests
import json
from serpapi import GoogleSearch




secrets_db = {"host": "<DB-Hostname>",
            "port": "<PORT>",
            "database": "nyt_import",
            "user": "<USER>",
            "pass": "<PW>"}


params_google = {
  "engine": "google_jobs",
  "q": "data scientist",
  "location": "Austria",
  "hl": "en",
  "api_key": "<API-Key>"
}


search = GoogleSearch(params_google)
results = search.get_dict()
jobs_results = results['jobs_results']






conn = psycopg2.connect(host=secrets['host'],
                        port=secrets['port'],
                        database=secrets['database'],
                        user=secrets["user"],
                        password=secrets["pass"])
cur = conn.cursor()



tableName = "ds21_B1_jobs"
sql1 =   "DROP TABLE IF EXISTS " + tableName + ";\n"
sql2 = "CREATE TABLE IF NOT EXISTS " + tableName + "\n"


sql3 =  '''  (id serial, 
            url VARCHAR(2048) UNIQUE, 
            status INTEGER,  
            body JSONB, 
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), 
            updated_at TIMESTAMPTZ);
        '''
        
sql =  sql2 + sql3
cur.execute(sql)


key = "?api-key=" + api_key["key"]
url = default_url + "/viewed/30.json" + key


response = requests.get(url)
text = response.text
status = response.status_code
js = response.json()

for i in js["results"]:
    del i["org_facet"]
    del i["per_facet"]
    del i["geo_facet"]
    del i["des_facet"]
    del i["abstract"]
    del i["media"]
    del i["column"]
    del i["eta_id"]

    t = i["url"]
    u = str(i).replace("'", "\"")
    sql = 'INSERT INTO ds21m031_nyt_mostViewed (url, status, body) VALUES  (%s, %s, %s)' 
    row = cur.execute(sql, (t, status, u))


print('Closing database connection...')
conn.commit()
cur.close()
