import importlib.metadata,subprocess,sys,torch
print('EXAMPLE 2 - FRANKA GEAR ASSEMBLY')
print('Python:',sys.version.split()[0])
for p in ['isaacsim','isaaclab','torch','rl-games']:
 try:print(p+':',importlib.metadata.version(p))
 except importlib.metadata.PackageNotFoundError:print(p+': install with ./workshop.sh setup')
print('CUDA available:',torch.cuda.is_available())
if not torch.cuda.is_available():raise SystemExit('A supported NVIDIA GPU is required.')
print('GPU:',torch.cuda.get_device_name(0))
subprocess.run(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader'],check=True)
print('DOCTOR PASSED')
