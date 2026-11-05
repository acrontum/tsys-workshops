"""Download the exact published baseline and install the pinned RL backend."""
import hashlib,importlib.metadata,importlib.util,os,subprocess,sys,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ASSET_BASE='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Factory/NIST/'
ASSET_HASHES={
 'gear_base.usd':'0c4353313c6915838ff442b77307957788310db698f1f592b022fa90f03e0d6d',
 'gear_medium.usd':'fafb31f5181da58d2b60c6e2f59370774bd95672c76a5cec51e12845a10982de',
 'gear_large.usd':'3181a9aef9abeb5c762e13e104dcf3774439935b9f799abd77a877c0e9ad8c57',
 'gear_small.usd':'c8cf5024af2b7290963a5d18a4612b998438064216bc8d6acce3c2cd31fed7d3',
}
(ROOT/'assets').mkdir(exist_ok=True)
for filename,expected in ASSET_HASHES.items():
 asset=ROOT/'assets'/filename
 if not asset.exists() or hashlib.file_digest(asset.open('rb'),'sha256').hexdigest()!=expected:
  temporary=asset.with_suffix('.download')
  urllib.request.urlretrieve(ASSET_BASE+filename,temporary)
  if hashlib.file_digest(temporary.open('rb'),'sha256').hexdigest()!=expected:
   raise RuntimeError('NIST asset checksum mismatch: '+filename)
  temporary.replace(asset)
 print('Verified NIST reference asset: '+filename,flush=True)
URL='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rl_games/Isaac-Factory-GearMesh-Direct-v0/checkpoint.pth'
SHA='c5096f3ea1f8e1fadb6e5546296c3db57a135e25ea2645a6effc2403d93401e1'
REV='5cf4f9ff07e9a01bbefd167cf19b6804b294b51c'
DEST=ROOT/'checkpoints/nvidia_factory_gear_pretrained.pth'
DEST.parent.mkdir(exist_ok=True)
if not DEST.exists() or hashlib.file_digest(DEST.open('rb'),'sha256').hexdigest()!=SHA:
 print('Downloading NVIDIA Factory gear policy (203 MiB).',flush=True)
 temp=DEST.with_suffix('.download')
 urllib.request.urlretrieve(URL,temp)
 if hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()!=SHA:raise RuntimeError('Checkpoint checksum mismatch; retry the download.')
 temp.replace(DEST)
print('Verified published checkpoint: '+SHA,flush=True)
runner_spec=importlib.util.find_spec('rl_games')
runner_file=Path(runner_spec.origin).parent/'torch_runner.py' if runner_spec else None
backend_matches=runner_file is not None and runner_file.exists() and hashlib.file_digest(runner_file.open('rb'),'sha256').hexdigest()=='661afc378758a8913b0e6b42fc66cf5b97de89e786c8f707f5915036da0b94c0'
if not backend_matches:
 archive=ROOT/'output/rl_games_pinned.zip';archive.parent.mkdir(exist_ok=True)
 urllib.request.urlretrieve('https://codeload.github.com/isaac-sim/rl_games/zip/'+REV,archive)
 uv=Path(sys.executable).parents[2]/'tools/bin/uv'
 executable=str(uv) if uv.exists() else 'uv'
 subprocess.run([executable,'pip','install','--python',sys.executable,str(archive),'gym','standard-distutils'],check=True)
print('SETUP COMPLETE: Franka Factory checkpoint and RL Games ready.',flush=True)
