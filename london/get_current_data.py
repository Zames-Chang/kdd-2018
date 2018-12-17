import requests
url = 'https://biendata.com/competition/airquality/ld/2018-04-19-0/2018-04-20-23/2k0d1d8'
respones= requests.get(url)
with open ("bj_meteorology_2018-04-19-0-2018-04-20-23.csv",'w') as f:
    f.write(respones.text)
