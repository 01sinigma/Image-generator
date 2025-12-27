"""Check available Qwen-related nodes in ComfyUI"""
import urllib.request
import json

r = urllib.request.urlopen('http://127.0.0.1:8188/object_info')
d = json.loads(r.read())

print("=" * 60)
print("Looking for Qwen-related nodes")
print("=" * 60)

# Search for qwen nodes
qwen_nodes = [k for k in d.keys() if 'qwen' in k.lower()]
print(f"\nNodes with 'qwen' in name: {len(qwen_nodes)}")
for n in qwen_nodes:
    print(f"  - {n}")
    inputs = d[n].get('input', {}).get('required', {})
    for inp_name, inp_info in inputs.items():
        val = inp_info[0] if isinstance(inp_info, list) else inp_info
        if isinstance(val, list) and len(val) > 3:
            val = val[:3] + ['...']
        print(f"      {inp_name}: {val}")

# Search for text encode nodes
print("\n" + "=" * 60)
print("Text Encoding nodes")
print("=" * 60)

text_nodes = [k for k in d.keys() if 'text' in k.lower() and 'encode' in k.lower()]
for n in text_nodes[:10]:
    print(f"  - {n}")
    inputs = d[n].get('input', {}).get('required', {})
    for inp_name, inp_info in list(inputs.items())[:3]:
        print(f"      {inp_name}")

# Check model loader types that work with qwen
print("\n" + "=" * 60)  
print("CLIP/Text loaders with qwen support")
print("=" * 60)

for node_name in ['CLIPLoaderGGUF', 'DualCLIPLoaderGGUF', 'CLIPLoader']:
    if node_name in d:
        inputs = d[node_name].get('input', {}).get('required', {})
        if 'type' in inputs:
            types = inputs['type'][0]
            if 'qwen' in str(types).lower():
                print(f"\n{node_name}:")
                print(f"  Types with qwen: {[t for t in types if 'qwen' in t.lower()]}")
                if 'clip_name' in inputs:
                    print(f"  Available clips: {inputs['clip_name'][0]}")

# Check sampler requirements
print("\n" + "=" * 60)
print("Checking ModelSamplingQwen node")
print("=" * 60)

sampling_nodes = [k for k in d.keys() if 'sampling' in k.lower() or 'sampl' in k.lower() and 'qwen' in k.lower()]
print(f"Found: {sampling_nodes}")

# Check for special Qwen model patching
qwen_patch = [k for k in d.keys() if 'model' in k.lower() and ('patch' in k.lower() or 'apply' in k.lower())]
print(f"\nModel patching nodes: {qwen_patch[:5]}...")

