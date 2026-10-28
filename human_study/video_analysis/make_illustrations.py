#!/usr/bin/env python3
"""Build traceable illustrations from the existing baseline recordings only."""
import argparse,hashlib,json
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
ROWS=[(101,0,[4.0,7.25,14.55,25.4],'A | Offset approach, then empty-hand retreat'),(111,85,[15.3,15.6,15.85,16.1],'B | Near-object contact displaces the toy'),(101,64,[15.6,17.55,19.5,21.45],'C | Visual completion: lift, carry, release into box')]
CLIPS=[(111,85,14.8,17.0,'contact_displacement.gif'),(101,64,16.8,21.45,'baseline_completion.gif'),(101,57,18.5,20.5,'placement_ambiguity.gif')]
def main():
 p=argparse.ArgumentParser();p.add_argument('--video-root',type=Path,required=True);args=p.parse_args();out=Path(__file__).resolve().parent
 font=ImageFont.truetype(FONT,15);fontsmall=ImageFont.truetype(FONT,13)
 def source(seed,trial):return args.video_root/f'human30_seed{seed}_eval42'/f'rollout-episode-{trial}.mp4'
 def frame(cap,t):
  cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*20));ok,f=cap.read();assert ok
  return Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)).crop((200,250,850,710))
 sheet=Image.new('RGB',(1536,1026),'white');d=ImageDraw.Draw(sheet)
 d.text((12,10),'Existing BASELINE recordings | All examples are human30 policies | Not a before/after video comparison',font=font,fill='black')
 metadata={'figure_rows':[],'clips':[]}
 for row,(seed,trial,times,title) in enumerate(ROWS):
  path=source(seed,trial);sha=hashlib.sha256(path.read_bytes()).hexdigest();cap=cv2.VideoCapture(str(path));top=44+row*326
  d.text((12,top),f'{title} | seed {seed}, trial {trial} | source SHA256 {sha[:12]}',font=font,fill='black')
  for col,t in enumerate(times):
   x=col*384;sheet.paste(frame(cap,t).resize((384,272)),(x,top+48));d.text((x+8,top+27),f't={t:.2f}s / frame {round(t*20)}',font=fontsmall,fill='black')
  cap.release();metadata['figure_rows'].append({'seed':seed,'trial':trial,'times_seconds':times,'source_sha256':sha})
 sheet.save(out/'baseline_visual_evidence.jpg',quality=91)
 for seed,trial,start,end,name in CLIPS:
  path=source(seed,trial);sha=hashlib.sha256(path.read_bytes()).hexdigest();cap=cv2.VideoCapture(str(path));frames=[];indices=list(range(round(start*20),round(end*20)+1,2))
  for index in indices:
   im=Image.new('RGB',(520,414),'white');im.paste(frame(cap,index/20).resize((520,368)),(0,46));draw=ImageDraw.Draw(im)
   draw.text((8,5),f'BASELINE | seed {seed}, trial {trial} | t={index/20:.2f}s',font=font,fill='black')
   draw.text((8,26),f'Source SHA256 {sha[:12]} | cropped, real-time 10 fps',font=fontsmall,fill='black');frames.append(im.convert('P', palette=Image.Palette.ADAPTIVE, colors=256))
  cap.release();frames[0].save(out/'clips'/name,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False,disposal=2)
  metadata['clips'].append({'seed':seed,'trial':trial,'source_sha256':sha,'frame_indices':indices,'start_seconds':indices[0]/20,'end_seconds':indices[-1]/20,'output_fps':10,'speed':1.0,'crop_xywh':[200,250,650,460],'path':'clips/'+name})
 (out/'illustration_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__':main()
