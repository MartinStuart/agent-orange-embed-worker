"""Build a ~30-row local sample (id, embed_text, main_image_url) from the pulled
offering CSVs. Local test only; never touches the DB."""
import csv, glob, json, re, uuid, html
from urllib.parse import urlparse

csv.field_size_limit(10**9)
FILES = sorted(glob.glob('/workspace/products/ccp_full/offering_ccp_full_part*.csv')) + \
        sorted(glob.glob('/workspace/weconnect_pull/offering_weconnect_0[123].csv'))

# (name substring, optional host requirement) - picked for variety + retrieval sanity checks
PICKS = [
    ('Sideways Sofa', None), ('Lounge Chair and Ottoman', None), ('Truss Coffee Table', None),
    ('Kaiser Idell', None), ('Luxe Bamboo Solid Rug', None), ('Coyuchi Pescadero Matelasse Organic Blanket', None),
    ('Linen Oversized Mens Shirt', None), ('Linen French Cuff Blouse', None), ('Silk Polka Dot Dress', None),
    ('Flare Sun Dress', None), ('Star Hoop Earrings', None), ('Pearl Threader Earrings', None),
    ('Ritual Mug', None), ('Kinto Cast Amber Mug', None), ('Navy Blue Google Pixel 8 Phone Case', None),
    ('Organic Dog Bed + Pillow Bolster', None), ('Plush Carrot Dog Toy', None), ('BRUNA Scented Candle', None),
    ('Lavender Bar Soap', None), ('tentree x Saye M89 Sneaker', None), ('YETI Wine Chiller', None),
    ('Weighted Blanket', None), ('Organic Latex Travel Neck Pillow', None), ('Chocolate Pistachio - Protein Bars', None),
    # non-Shopify image hosts
    (None, 'aignerchocolates.com'), (None, 'www.aromesoil.com'), (None, 'cdn.epiphanyglass.com'),
    (None, 'bodegasmaximoabete.com'), (None, 'i0.wp.com'),
]

def first_img(s):
    if not s or s == '{}':
        return ''
    return s.strip('{}').split(',')[0].strip('"')

def strip_html(s):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', s or ''))).strip()

def embed_text(x):
    try:
        a = json.loads(x['attributes'] or '{}')
    except Exception:
        a = {}
    tags = a.get('tags') or []
    parts = [x['name'], a.get('vendor') or '', x['category'] or '', ', '.join(tags[:15]) if isinstance(tags, list) else '',
             strip_html(x['description'])[:1500]]
    return ' | '.join(p for p in parts if p)

found = {}
for f in FILES:
    for x in csv.DictReader(open(f, encoding='utf-8', newline='')):
        img = first_img(x['image_urls']); host = urlparse(img).netloc
        for i, (name, h) in enumerate(PICKS):
            if i in found:
                continue
            if name and name.lower() in x['name'].lower() and img:
                found[i] = (x, img)
            elif h and host == h and len(x['description'] or '') > 40:
                found[i] = (x, img)
    if len(found) == len(PICKS):
        break

rows = []
for i in sorted(found):
    x, img = found[i]
    rows.append({'id': str(uuid.uuid5(uuid.NAMESPACE_URL, x['url'] or x['name'])), 'embed_text': embed_text(x), 'main_image_url': img})
# one row with a missing image (copy of a real product's text, image stripped)
x, _ = found[0]
missing = dict(rows[0]); missing['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'missing-image-test')); missing['main_image_url'] = None
missing['embed_text'] = embed_text(found[12][0])  # Ritual Mug text, no image
rows.append(missing)
with open('/workspace/embed-worker/sample/sample.jsonl', 'w') as fh:
    for r in rows:
        fh.write(json.dumps(r, ensure_ascii=False) + '\n')
print(len(rows), 'rows;', sum(1 for r in rows if r['main_image_url'] and 'cdn.shopify.com' not in r['main_image_url']), 'non-shopify;',
      sum(1 for r in rows if not r['main_image_url']), 'missing')
for r in rows:
    print(r['id'][:8], (r['main_image_url'] or 'NULL')[:70], '|', r['embed_text'][:70])
