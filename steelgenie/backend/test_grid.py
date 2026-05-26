import requests

r = requests.post("http://localhost:8000/analyse",
    json={"filename": "sample img.png", "page_index": 0, "scale_ratio": 96},
    timeout=180)
data = r.json()
members = data.get("members", [])
beams = [m for m in members if m["type"] == "beam"]

has_line = [b for b in beams if b.get("bx1") is not None]
no_line  = [b for b in beams if b.get("bx1") is None]

print(f"Total beams: {len(beams)}")
print(f"  With span endpoints (bx1 not None): {len(has_line)}")
print(f"  Without span endpoints (bx1=None) : {len(no_line)}")
print()

if no_line:
    print("Beams WITHOUT endpoints (no line will be drawn):")
    for b in no_line:
        print(f"  {b['profile']:14s}  dir={b.get('beam_dir')}  x={b['x']:.3f} y={b['y']:.3f}")

if has_line:
    print()
    print("Beams WITH endpoints (line WILL be drawn):")
    for b in has_line:
        print(f"  {b['profile']:14s}  dir={b.get('beam_dir')}  "
              f"({b['bx1']:.3f},{b['by1']:.3f})->({b['bx2']:.3f},{b['by2']:.3f})")
