"""Acerto dos especialistas (café e soja v2) sob cada degradação/nível -> define o que é 'inviável'."""
import os, sys, io, json, numpy as np
sys.path.insert(0, '/home/kaua-matheus/Documents/Agroscope_IA/Steps')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from utils.inference import Diagnoser
from degrade import LEVELS, apply

M = '/home/kaua-matheus/Documents/Agroscope_IA/Steps/Models/Saved/Expert'
idx = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache/index.json')))
A = '/media/kaua-matheus/HD3201/Data/Agroscope'
out = {}
for culture, group, model, test_dir in [('coffee', 'coffee_test', f'{M}/Coffee/coffee.pth', f'{A}/Coffee/Test_Coffee'),
                                        ('soybean', 'soy_test', f'{M}/Soybean/soybean.pth', f'{A}/Soybean/Test_Soybean')]:
    d = Diagnoser(model)
    classes = d.primary.classes
    # rótulo verdadeiro: casa o arquivo do cache com a ordem de files() do build_cache (mesma ordem ordenada)
    srcs = []
    for root, _, fs in os.walk(test_dir):
        srcs += [(os.path.join(root, f), os.path.basename(root)) for f in fs if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    srcs.sort()
    items = [x for x in idx if x[2] == group]
    items.sort(key=lambda x: x[0])
    assert len(items) == len(srcs), (len(items), len(srcs))
    labels = [c for _, c in srcs]
    imgs = [Image.open(x[0]).convert('RGB') for x in items]
    rng = np.random.default_rng(0)
    for kind in LEVELS:
        for lvl in range(4):
            if lvl == 0 and kind != 'gauss': continue
            np.random.seed(1)
            ok = 0; conf = []
            for im, y in zip(imgs, labels):
                b = io.BytesIO(); apply(im, kind, lvl).save(b, 'PNG'); b.seek(0)
                r = d.predict(Image.open(b), tta=False)
                ok += r['final']['predicted_class'] == y; conf.append(r['final']['confidence'])
            key = 'clean' if lvl == 0 else f'{kind}_{lvl}'
            out.setdefault(key, {})[culture] = round(ok / len(imgs), 4)
            print(culture, key, round(ok / len(imgs), 4), 'conf_media', round(float(np.mean(conf)), 3), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expert_impact.json'), 'w'), indent=1)
