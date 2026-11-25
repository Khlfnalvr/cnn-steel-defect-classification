"""Gabungkan template, skrip, dan data model menjadi satu berkas HTML: dist/neu-steel-defect.html

    python web/build.py

Data (web/data/*.json) dibuat oleh src/export_web.py.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
read = lambda *p: open(os.path.join(HERE, *p), encoding='utf-8').read()

html = (read('template.html')
        .replace('__ENGINE__', read('engine.js'))
        .replace('__SCENE__', read('scene.js'))
        .replace('__S3UI__', read('s3ui.js'))
        .replace('__PAGEJS__', read('page.js'))
        .replace('__WEIGHTS__', read('data', 'weights.json'))
        .replace('__SAMPLES__', read('data', 'samples.json'))
        .replace('__META__', read('data', 'meta.json')))

out_dir = os.path.join(HERE, '..', 'dist')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, 'neu-steel-defect.html')
open(out, 'w', encoding='utf-8').write(html)
print('ok', len(html) // 1024, 'KB ->', os.path.normpath(out))
