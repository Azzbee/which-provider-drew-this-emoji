"""Jev as the final decision-maker over three experts: DINOv2, ConvNeXt and the file-byte model.

Jev is consulted only on the development images where the three experts are NOT unanimous
(205 of 1,976). Everywhere else the unanimous answer is kept. Three ways of telling Jev
what the experts think are compared, each with the query image and one labelled reference
image per provider:

  onehot     each expert's answer as a one-hot vector (no confidences)
  agreement  only the labels the experts proposed, with who proposed each; Jev must choose
             among those proposals (the choice list is restricted to them)
  probs      each expert's full seven-class probability distribution

Commands
  python jev_three_experts.py prepare      # builds images/ and routed.json (no API, no key)
  python jev_three_experts.py dryrun       # builds and validates every payload (no API, no key)
  python jev_three_experts.py run          # paid: needs TYPESAFE_API_KEY in YOUR shell; capped at $1
  python jev_three_experts.py evaluate     # scores whatever answers are cached (no API)

Cost: 205 images x 3 modes = 615 requests. On 18 October, 324 comparable requests cost $0.035,
so expect roughly $0.07. The client refuses to exceed the --budget cap and never retries.
Set the key yourself (export TYPESAFE_API_KEY=...); it is never written to disk or printed.
"""
import argparse, base64, hashlib, io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CLASSES = ['apple', 'facebook', 'google', 'messenger', 'mozilla', 'samsung', 'whatsapp']
MODEL = 'jev-1.14.0'
EXPERTS = ('dinov2', 'convnext', 'file_bytes')
# Same seven medoid references as the 18 October Jev experiment (all from the training split).
REFERENCES = {'apple': '03054', 'facebook': '09572', 'google': '08730', 'messenger': '00187',
              'mozilla': '05289', 'samsung': '09766', 'whatsapp': '08098'}
MODES = ('onehot', 'agreement', 'probs')

COMMON = ("Identify which provider rendered the query emoji image. Classify the visual rendering "
          "style/provider, not the emoji's meaning. The seven labelled reference images are examples "
          "from the training data; they may depict different emoji contents and are not exhaustive. "
          "Facebook and Messenger are distinct dataset labels. ")
PROMPTS = {
    'onehot': COMMON + ("The state gives the answers of three fallible experts as one-hot vectors over the "
                        "seven providers: dinov2 and convnext look at the pixels; file_bytes looks only at how "
                        "the PNG file was encoded. They disagree on this image. Weigh their answers together "
                        "with the image and references, and select exactly one provider."),
    'agreement': COMMON + ("Three fallible experts disagree on this image: dinov2 and convnext look at the pixels; "
                           "file_bytes looks only at how the PNG file was encoded. The state lists each provider "
                           "they proposed and which experts proposed it. Choose the proposal best supported by "
                           "the image, the references and the experts' agreement."),
    'probs': COMMON + ("The state gives the full seven-class probability distributions of three fallible experts: "
                       "dinov2 and convnext look at the pixels; file_bytes looks only at how the PNG file was "
                       "encoded. They disagree on this image. Treat them as evidence together with the image and "
                       "references, and select exactly one provider."),
}


def transform_image(source):
    """Identical to the 18 October pipeline: alpha onto white, Lanczos contain, 280x280 letterbox."""
    with Image.open(source) as src:
        rgba = ImageOps.exif_transpose(src).convert('RGBA')
        rgb = Image.alpha_composite(Image.new('RGBA', rgba.size, (255, 255, 255, 255)), rgba).convert('RGB')
        small = ImageOps.contain(rgb, (280, 280), Image.Resampling.LANCZOS)
        out = Image.new('RGB', (280, 280), 'white')
        out.paste(small, ((280 - small.width) // 2, (280 - small.height) // 2))
        s = io.BytesIO(); out.save(s, format='PNG', optimize=False)
        return s.getvalue()


def experts():
    E = np.load(HERE / 'experts_dev.npz')
    return E['D'], E['C'], E['F'], E['y'], [str(i) for i in E['ids']]


def prepare(args):
    D, C, F, y, ids = experts()
    a = [P.argmax(1) for P in (D, C, F)]
    routed = [i for i in range(len(y)) if not (a[0][i] == a[1][i] == a[2][i])]
    img = HERE / 'images'; img.mkdir(exist_ok=True)
    train = Path(args.data) / 'train'
    for cls, rid in REFERENCES.items():
        (img / f'ref_{cls}.png').write_bytes(transform_image(train / f'{rid}.png'))
    for i in routed:
        (img / f'{ids[i]}.png').write_bytes(transform_image(train / f'{ids[i]}.png'))
    json.dump({'routed_rows': routed, 'routed_ids': [ids[i] for i in routed]}, open(HERE / 'routed.json', 'w'))
    print(f'{len(routed)} routed images prepared in {img}')


def asset(path):
    return {'type': 'image', 'format': 'png', 'data_base64': base64.b64encode(Path(path).read_bytes()).decode()}


def payload(mode, img_id, probs):
    assets = {'asset_query': asset(HERE / 'images' / f'{img_id}.png')}
    refs = []
    for k, cls in enumerate(CLASSES):
        assets[f'asset_ref_{k}'] = asset(HERE / 'images' / f'ref_{cls}.png')
        refs.append({'provider': cls, 'image': f'asset_ref_{k}'})
    state = {'query_image': 'asset_query', 'training_references': refs}
    labels = {e: CLASSES[int(np.argmax(p))] for e, p in zip(EXPERTS, probs)}
    options = CLASSES
    if mode == 'onehot':
        state['expert_one_hot'] = {e: {c: int(c == labels[e]) for c in CLASSES} for e in EXPERTS}
    elif mode == 'agreement':
        proposals = {}
        for e in EXPERTS: proposals.setdefault(labels[e], []).append(e)
        state['proposals'] = [{'provider': c, 'proposed_by': v, 'votes': len(v)} for c, v in proposals.items()]
        options = [c for c in CLASSES if c in proposals]
    elif mode == 'probs':
        state['expert_probabilities'] = {e: {c: round(float(v), 6) for c, v in zip(CLASSES, p)} for e, p in zip(EXPERTS, probs)}
    q = {'type': 'choice', 'instructions': PROMPTS[mode],
         'criteria': {c: f'The query emoji was rendered by the {c} provider.' for c in options}}
    return {'model': MODEL, 'state': state, 'assets': assets, 'questions': {'provider': q}}


def iterate():
    D, C, F, y, ids = experts()
    r = json.load(open(HERE / 'routed.json'))
    for i in r['routed_rows']:
        yield i, ids[i], (D[i], C[i], F[i])


def dryrun(args):
    import jev_client
    n = 0
    for mode in MODES:
        for i, img_id, probs in iterate():
            p = payload(mode, img_id, probs)
            jev_client._validate_payload(p)
            assert 'label' not in json.dumps(p['state']).lower()
            n += 1
    print(f'{n} payloads built and validated; nothing sent')


def answers_path(mode):
    return HERE / 'answers' / f'{mode}.json'


def run(args):
    import jev_client
    client = jev_client.BudgetedJevClient(HERE / 'jev_cache', args.budget)
    for mode in args.modes:
        path = answers_path(mode); path.parent.mkdir(exist_ok=True)
        got = json.load(open(path)) if path.exists() else {}
        for i, img_id, probs in iterate():
            if img_id in got: continue
            p = payload(mode, img_id, probs)
            raw = client.run(p, request_key=f'three-experts:{mode}:{img_id}')
            got[img_id] = raw['answers']['provider']
            json.dump(got, open(path, 'w'), indent=0)
        print(mode, len(got), 'answers', client.budget_summary())


def evaluate(args):
    D, C, F, y, ids = experts()
    eps = 1e-6
    ref = ((D + C) / 2).argmax(1)
    geo = np.exp(np.log(D + eps) + np.log(C + eps) + np.log(F + eps)).argmax(1)
    a = [P.argmax(1) for P in (D, C, F)]
    routed = json.load(open(HERE / 'routed.json'))['routed_rows']
    rows = [('DINOv2 + ConvNeXt equal average (reference)', ref),
            ('Three experts, geometric mean (no Jev)', geo)]
    oracle = geo.copy()
    for i in routed:
        if y[i] in (a[0][i], a[1][i], a[2][i]): oracle[i] = y[i]
    for mode in MODES:
        path = answers_path(mode)
        if not path.exists(): continue
        got = json.load(open(path))
        if len(got) < len(routed):
            print(f'{mode}: {len(got)}/{len(routed)} answers, incomplete, not scored'); continue
        p = geo.copy()   # unanimous rows: all three agree, so geo = the unanimous label
        for i in routed: p[i] = CLASSES.index(got[ids[i]]['choice'])
        rows.append((f'Jev decides the {len(routed)} non-unanimous images ({mode})', p))
    rows.append(('Oracle: right whenever any expert is right (upper bound)', oracle))
    out = []
    for name, p in rows:
        fixed = int(((p == y) & (ref != y)).sum()); broke = int(((p != y) & (ref == y)).sum())
        out.append({'name': name, 'correct': int((p == y).sum()), 'accuracy': round(float((p == y).mean() * 100), 2),
                    'fixed_vs_reference': fixed, 'new_errors_vs_reference': broke})
        print(f'{(p == y).mean() * 100:6.2f}%  {(p == y).sum():5d}  +{fixed:<3d} -{broke:<3d} {name}')
    json.dump(out, open(HERE / 'jev_results.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('prepare'); s.add_argument('--data', default='../../../codebase/data/extracted')
    sub.add_parser('dryrun')
    s = sub.add_parser('run'); s.add_argument('--budget', default='1'); s.add_argument('--modes', nargs='+', default=list(MODES))
    sub.add_parser('evaluate')
    a = ap.parse_args()
    {'prepare': prepare, 'dryrun': dryrun, 'run': run, 'evaluate': evaluate}[a.cmd](a)
