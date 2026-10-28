#!/usr/bin/env python3
"""Extract predeclared baseline video samples; does not run the simulator."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

TRIALS = [0, 7, 14, 21, 28, 35, 42, 50, 57, 64, 71, 78, 85, 92, 99]
SEEDS = [101, 111]

def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--video-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'contact_sheets').mkdir(exist_ok=True)
    records=[]
    for seed in SEEDS:
        for trial in TRIALS:
            source=args.video_root/f'human30_seed{seed}_eval42'/f'rollout-episode-{trial}.mp4'
            cap=cv2.VideoCapture(str(source)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS); cap.release()
            indices=[max(1,round(i*(n-1)/11)) for i in range(12)]
            filt='select='+ '+'.join(f'eq(n\\,{i})' for i in indices) + ',crop=650:460:200:250,scale=384:272'
            cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(source),'-vf',filt,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
            raw=subprocess.check_output(cmd); frames=np.frombuffer(raw,np.uint8).reshape(-1,272,384,3)
            assert len(frames)==12,(source,len(frames))
            sheet=Image.new('RGB',(1536,926),'white'); draw=ImageDraw.Draw(sheet)
            draw.text((12,10),f'BASELINE seed {seed} | trial {trial} | crop x=200..850, y=250..710 | recorded run, not rerun labels',fill='black')
            for i,(frame,index) in enumerate(zip(frames,indices)):
                x=(i%4)*384;y=32+(i//4)*298
                sheet.paste(Image.fromarray(frame),(x,y+20));draw.text((x+6,y+3),f't={index/fps:.2f}s | frame {index}',fill='black')
            name=f'contact_sheets/seed{seed}_trial{trial:02d}.jpg';sheet.save(args.output/name,quality=86)
            records.append({'training_seed':seed,'trial':trial,'source_video':str(source),'source_sha256':sha256(source),'fps':fps,'n_frames':n,'duration_seconds':n/fps,'sample_frame_indices':indices,'sample_times_seconds':[i/fps for i in indices],'contact_sheet':name,'official_success_label':None})
    meta={'selection_rule':'Before viewing the new samples: 15 approximately equally spaced trial IDs per training seed; [0,7,14,21,28,35,42,50,57,64,71,78,85,92,99]. 30 video files. This is a systematic qualitative sample, not a random estimate of population failure frequencies.','analysis_date':'2026-10-29','video_root_scope':'Only two baseline recording directories found; no final Mimic policy recording is available.','run_separation':'These are the original video-enabled baseline evaluations. The formal recorded stdout/JSONL evaluations ran later with video=false. Trial numbers are NOT joined across runs.','crop_xywh':[200,250,650,460],'trials':records}
    (args.output/'source_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'videos':len(records),'unique_video_hashes':len(set(r['source_sha256'] for r in records)),'sample_ids':TRIALS}))

if __name__=='__main__':main()
