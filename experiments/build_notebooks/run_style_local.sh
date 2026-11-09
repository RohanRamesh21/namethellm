#!/bin/bash
# download the three small style embedders with resumable curl, then run the experiment script
set -u
cd /home/rohan/kagglecomp
for repo in AnnaWegmann/Style-Embedding gabrielloiseau/LUAR-MUD-sentence-transformers gabrielloiseau/LUAR-CRUD-sentence-transformers; do
  dir=/tmp/models/${repo//\//--}; mkdir -p "$dir"
  python3 -I - "$repo" > "$dir/.files" <<'PY'
import json,sys,urllib.request
info=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{sys.argv[1]}?blobs=true',timeout=60))
files={s['rfilename']:s.get('size') for s in info['siblings'] if '/' not in s['rfilename']}
st=any(n.endswith('.safetensors') for n in files)
for n,sz in files.items():
    if n.endswith(('.json','.txt','.model','.safetensors')) or (n.endswith('.bin') and not st): print(n,sz)
PY
  while read -r name size; do
    for try in 1 2 3 4 5 6 7 8; do
      have=$(stat -c %s "$dir/$name" 2>/dev/null || echo 0)
      [ "$have" = "$size" ] && break
      curl -sL -C - --retry 3 -o "$dir/$name" "https://huggingface.co/$repo/resolve/main/$name"
    done
    echo "$repo $name $(stat -c %s "$dir/$name") / $size"
  done < "$dir/.files"
done
echo DOWNLOADS_DONE
python3 experiments/style_embedders.py --data name-that-llm --out /tmp/style_out --models /tmp/models \
  --candidates wegmann,luar_mud,luar_crud --local qwen_emb=cached --local bert=cached
echo RUN_DONE
