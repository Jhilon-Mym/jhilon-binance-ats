import urllib.request, json
url='http://127.0.0.1:8000/api/login'
payload=json.dumps({'email':'jewelmym2022@gmail.com','password':'Jewel1988'}).encode('utf-8')
req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        print('STATUS', r.status)
        print(r.read().decode())
except Exception as e:
    print('ERR', e)
