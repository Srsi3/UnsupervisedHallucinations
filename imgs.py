import os, requests

API_KEY = 'YOUR_NASA_API_KEY'
SAVE_DIR = 'nasa_galaxy_images'
os.makedirs(SAVE_DIR, exist_ok=True)

def query_images(query, page=1):
    url = 'https://images-api.nasa.gov/search'
    params = {'q': query, 'media_type': 'image', 'page': page}
    r = requests.get(url, params=params)
    return r.json().get('collection', {}).get('items', [])

def download_highres(item):
    links = item.get('links', [])
    for link in links:
        href = link.get('href', '')
        if href.lower().endswith('~orig.jpg') or href.lower().endswith('.jpg'):
            fname = os.path.join(SAVE_DIR, os.path.basename(href))
            if not os.path.exists(fname):
                img = requests.get(href).content
                with open(fname, 'wb') as f: f.write(img)
            break

for page in range(1, 51):
    results = query_images('galaxy', page=page)
    if not results: break
    for item in results:
        download_highres(item)
