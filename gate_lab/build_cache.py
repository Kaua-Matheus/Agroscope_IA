"""
Monta o conjunto do modelo de triagem (folha / não-folha), já reduzido para 768px.

Divisão pensada para medir generalização:
  - folhas de café, soja e tomate: treino/val com imagens das bases; teste com os conjuntos externos
  - trigo: só teste (cultura nunca vista pelo modelo)
  - Caltech-101: categorias de teste NUNCA aparecem no treino; categorias de planta (bonsai, lótus...) só no teste
  - trigo FusariumFootRot/BlackPoint (raiz/grão): só no teste, como "planta mas não folha"
"""
import os, sys, json, random
from multiprocessing import Pool
from PIL import Image, ImageOps

random.seed(0)
A = '/media/kaua-matheus/HD3201/Data/Agroscope'
# Caltech-101 (https://data.caltech.edu/records/mzrjq-6wc02) extraído; caminho via variável NEG_PATH
NEG = os.getenv('NEG_PATH', '/media/kaua-matheus/HD3201/Data/Agroscope/Negatives/101_ObjectCategories')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

def files(d, recursive=True):
    out = []
    for root, _, fs in os.walk(d):
        out += [os.path.join(root, f) for f in fs if f.lower().endswith(EXT)]
        if not recursive: break
    return sorted(out)

def sample(lst, n):
    lst = lst[:]; random.shuffle(lst); return lst[:n]

S = []  # (src, label, group, split)
def add(paths, label, group, split):
    S.extend((p, label, group, split) for p in paths)

# ---------- Folhas ----------
cof_test = set(os.listdir(f'{A}/Coffee/Test_Coffee/Healthy') + os.listdir(f'{A}/Coffee/Test_Coffee/Phoma') + os.listdir(f'{A}/Coffee/Test_Coffee/Rust'))
add(sample(files(f'{A}/Coffee/Base_Coffee'), 700), 'leaf', 'coffee_closeup', 'train')
add(sample([p for p in files(f'{A}/Coffee/Coffee_5') if os.path.basename(p) not in cof_test], 500), 'leaf', 'coffee_whole', 'train')
c1 = sample(files(f'{A}/Coffee/Coffee_1'), 600)
add(c1[:450], 'leaf', 'coffee_field', 'train'); add(c1[450:], 'leaf', 'coffee_field', 'test')
add(files(f'{A}/Coffee/Test_Coffee'), 'leaf', 'coffee_test', 'test')
soy = sample(files(f'{A}/Soybean/Base_Soybean'), 900)
add(soy[:700], 'leaf', 'soy_base', 'train')
add(sample(files(f'{A}/Soybean/Soybean_4'), 500), 'leaf', 'soy_other', 'train')
add(files(f'{A}/Soybean/Test_Soybean'), 'leaf', 'soy_test', 'test')
tom = sample(files(f'{A}/Tomato/Base_Tomato'), 900)
add(tom[:700], 'leaf', 'tomato_base', 'train'); add(tom[700:], 'leaf', 'tomato_test', 'test')
W = f'{A}/Wheat/Wheat_2/Original Dataset'
add(files(f'{W}/HealthyLeaf') + files(f'{W}/LeafBlight'), 'leaf', 'wheat_unseen', 'test')
add(files(f'{W}/FusariumFootRot') + files(f'{W}/BlackPoint'), 'not_leaf', 'wheat_root_grain', 'test')

# ---------- Não-folhas ----------
cats = sorted(c for c in os.listdir(NEG) if os.path.isdir(f'{NEG}/{c}'))
plant = {'bonsai', 'lotus', 'sunflower', 'water_lilly', 'strawberry', 'joshua_tree'}
objs = [c for c in cats if c not in plant and c != 'BACKGROUND_Google']
random.shuffle(objs)
n_test = int(0.3 * len(objs))
for c in objs[:n_test]:
    add(sample(files(f'{NEG}/{c}'), 40), 'not_leaf', 'objects_unseen', 'test')
for c in objs[n_test:]:
    add(sample(files(f'{NEG}/{c}'), 60), 'not_leaf', 'objects', 'train')
bg = sample(files(f'{NEG}/BACKGROUND_Google'), 1000)
add(bg[:230], 'not_leaf', 'background', 'train'); add(bg[230:], 'not_leaf', 'background', 'test')
for c in plant:
    add(files(f'{NEG}/{c}'), 'not_leaf', 'plants_not_leaf', 'test')

def work(item):
    i, (src, label, group, split) = item
    dst = f'{OUT}/{split}/{label}/{group}_{i:05d}.jpg'
    try:
        with open(src, 'rb') as f:
            im = Image.open(f); im.draft('RGB', (768, 768)); im = ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((768, 768))
        os.makedirs(os.path.dirname(dst), exist_ok=True); im.save(dst, quality=92)
        return (dst, label, group, split)
    except Exception as e:
        return None

if __name__ == '__main__':
    with Pool(10) as p:
        res = [r for r in p.map(work, list(enumerate(S)), chunksize=16) if r]
    json.dump(res, open(f'{OUT}/index.json', 'w'))
    from collections import Counter
    print(len(S), 'ok', len(res))
    for k, v in sorted(Counter((r[3], r[1], r[2]) for r in res).items()): print(k, v)
