"""Check ComfyUI error details"""
import urllib.request
import json

try:
    response = urllib.request.urlopen('http://127.0.0.1:8188/history')
    history = json.loads(response.read())
    
    print("=" * 60)
    print("ComfyUI Error Analysis")
    print("=" * 60)
    
    for prompt_id, data in list(history.items())[-3:]:
        print(f"\nPrompt ID: {prompt_id}")
        
        status = data.get('status', {})
        print(f"  Status: {status.get('status_str', 'unknown')}")
        print(f"  Completed: {status.get('completed', False)}")
        
        # Messages
        messages = status.get('messages', [])
        if messages:
            print("  Messages:")
            for msg in messages:
                print(f"    - {msg}")
        
        # Exception messages if any
        if 'exception_message' in data:
            print(f"  Exception: {data['exception_message']}")
        
        # Check prompt data for errors
        prompt_data = data.get('prompt', [])
        if len(prompt_data) >= 3:
            workflow = prompt_data[2]
            print(f"  Workflow nodes: {list(workflow.keys())[:5]}...")
            
except Exception as e:
    print(f"Error: {e}")

# Check queue
print("\n" + "=" * 60)
print("Current Queue")
print("=" * 60)

try:
    response = urllib.request.urlopen('http://127.0.0.1:8188/queue')
    queue = json.loads(response.read())
    print(f"  Running: {len(queue.get('queue_running', []))}")
    print(f"  Pending: {len(queue.get('queue_pending', []))}")
except Exception as e:
    print(f"Error: {e}")

# Check object_info for available samplers  
print("\n" + "=" * 60)
print("Available Samplers")
print("=" * 60)

try:
    response = urllib.request.urlopen('http://127.0.0.1:8188/object_info')
    obj_info = json.loads(response.read())
    
    if 'KSampler' in obj_info:
        ks = obj_info['KSampler']
        inputs = ks.get('input', {}).get('required', {})
        
        # Show available options
        if 'sampler_name' in inputs:
            print(f"  Samplers: {inputs['sampler_name'][0][:5]}...")
        if 'scheduler' in inputs:
            print(f"  Schedulers: {inputs['scheduler'][0][:5]}...")
    
except Exception as e:
    print(f"Error: {e}")

# Check available VAE models
print("\n" + "=" * 60)
print("Available VAE Models")
print("=" * 60)

try:
    response = urllib.request.urlopen('http://127.0.0.1:8188/object_info')
    obj_info = json.loads(response.read())
    
    if 'VAELoader' in obj_info:
        vae_loader = obj_info['VAELoader']
        inputs = vae_loader.get('input', {}).get('required', {})
        
        if 'vae_name' in inputs:
            vae_names = inputs['vae_name'][0]
            print(f"  Available VAEs ({len(vae_names)}):")
            for v in vae_names[:10]:
                print(f"    - {v}")
    
except Exception as e:
    print(f"Error: {e}")

