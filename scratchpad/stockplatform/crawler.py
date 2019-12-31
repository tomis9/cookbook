import urllib.request
import zipfile
import os

# download and unzip data
url = "http://bossa.pl/pub/newconnect/mstock/mstncn.zip"
urllib.request.urlretrieve(url, 'data.zip')
zip_ref = zipfile.ZipFile('data.zip', 'r')
zip_ref.extractall('data')
zip_ref.close()
os.remove('data.zip')

# clean data
try:
    os.remove('./data/metancn.lst')
except OSError as e:
    print(e.strerror)

files = [os.path.join('data', filename) for filename in os.listdir('./data')
         if ".mst" in filename]

for filename in files:
    with open(filename) as f:
        d = f.readlines()
    if len(d):
        d[0] = d[0].replace('<', '').replace('>', '')
        csv_path = os.path.splitext(filename)[0] + '.csv'
        with open(csv_path, 'w') as f:
            f.writelines(d)
    os.remove(filename)
