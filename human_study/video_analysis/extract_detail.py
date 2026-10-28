#!/usr/bin/env python3
"""Dense follow-up frames for events unresolved in the 30-video first pass."""
import argparse,json,subprocess
from pathlib import Path
import imageio_ffmpeg,numpy as np
from PIL import Image,ImageDraw
SPECS=[(101,57,16,21),(101,85,10,15),(101,99,10,15),(111,85,14,19),(111,21,0,5)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--video-root',type=Path,required=True);args=p.parse_args()
 output=Path(__file__).resolve().parent;records=[]
 for seed,trial,start,end in SPECS:
  source=args.video_root/f'human30_seed{seed}_eval42'/f'rollout-episode-{trial}.mp4'
  times=np.linspace(max(.05,start),end,20);indices=[round(t*20) for t in times]
  vf='select='+ '+'.join(f'eq(n\\,{i})' for i in indices)+',crop=650:460:200:250,scale=384:272'
  raw=subprocess.check_output([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(source),'-vf',vf,'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
  frames=np.frombuffer(raw,np.uint8).reshape(-1,272,384,3);assert len(frames)==20
  sheet=Image.new('RGB',(1536,1522),'white');draw=ImageDraw.Draw(sheet);draw.text((12,10),f'BASELINE seed {seed} trial {trial} | DENSE FOLLOW-UP | {start}..{end}s',fill='black')
  for j,(frame,index) in enumerate(zip(frames,indices)):
   x=j%4*384;y=32+j//4*298;sheet.paste(Image.fromarray(frame),(x,y+20));draw.text((x+6,y+3),f't={index/20:.2f}s | frame {index}',fill='black')
  name=f'contact_sheets/seed{seed}_trial{trial:02d}_detail.jpg';sheet.save(output/name,quality=88)
  records.append({'training_seed':seed,'trial':trial,'frame_indices':indices,'times_seconds':[i/20 for i in indices],'contact_sheet':name,'reason':'Clarify initial approach failure or transient lift/contact and object disappearance in first-pass frames.'})
 # Native-rate / enlarged follow-up for object loss and the occluded release.
 for seed,trial,times in [(101,57,[19.7,19.95,20.05,20.15,20.25,20.35]),(101,99,[11.55,11.6,11.65,11.7,11.75,11.8])]:
  import cv2
  cap=cv2.VideoCapture(str(args.video_root/f'human30_seed{seed}_eval42'/f'rollout-episode-{trial}.mp4'))
  sheet=Image.new('RGB',(1950,980),'white');draw=ImageDraw.Draw(sheet)
  for j,t in enumerate(times):
   cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*20));ok,frame=cap.read();assert ok
   im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).crop((200,250,850,710));x=j%3*650;y=j//3*490
   sheet.paste(im,(x,y+30));draw.text((x+10,y+7),f'seed{seed} trial{trial} t={t:.2f}s',fill='black')
  cap.release();name=f'contact_sheets/seed{seed}_trial{trial:02d}_fine.jpg';sheet.save(output/name,quality=94)
  records.append({'training_seed':seed,'trial':trial,'frame_indices':[round(t*20) for t in times],'times_seconds':times,'contact_sheet':name,'reason':'Resolve fast contact displacement or release behind the box rim; retained ambiguity if the final object location is occluded.'})
 (output/'detail_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
