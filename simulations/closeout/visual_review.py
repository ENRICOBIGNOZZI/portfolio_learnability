"""Render final PDFs for human visual QA; programmatic geometry is additional."""
import json
from pathlib import Path
import fitz
from PIL import Image,ImageDraw
from simulations.closeout.results import DEST
from simulations.extended.design import ROOT
from simulations.provenance import file_hash,json_write


def render():
    folder=ROOT/'tmp/simulation_closeout_qa';folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for stem,path in [('simulation',DEST/'paper/simulation_closeout.pdf'),('manuscript',DEST/'paper/paper_staging.pdf'),('theory',DEST/'sources/Portfolio_Source_Regularity_Theory.pdf')]:
        doc=fitz.open(path);thumbs=[];outliers=[]
        for i,page in enumerate(doc):
            # Preserve 1000px images for per-page review in addition to contacts.
            pix=page.get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False)
            output=folder/f'{stem}_{i+1:03d}.png';pix.save(output)
            im=Image.open(output).convert('RGB');im.thumbnail((370,495))
            canvas=Image.new('RGB',(390,525),'#eee');canvas.paste(im,((390-im.width)//2,4))
            ImageDraw.Draw(canvas).text((12,508),f'{stem}, page {i+1}',fill='black');thumbs.append(canvas)
            outside=[]
            for block in page.get_text('blocks'):
                if block[0]<0 or block[1]<0 or block[2]>page.rect.width+.5 or block[3]>page.rect.height+.5:outside.append(list(block[:4]))
            if outside:outliers.append(dict(page=i+1,boxes=outside))
        contacts=[]
        for start in range(0,len(thumbs),12):
            sheet=Image.new('RGB',(390*3,525*4),'white')
            for j,t in enumerate(thumbs[start:start+12]):sheet.paste(t,((j%3)*390,(j//3)*525))
            p=folder/f'{stem}_contact_{start//12+1:02d}.png';sheet.save(p);contacts.append(str(p))
        records.append(dict(path=str(path.relative_to(DEST)),sha256=file_hash(path),pages=len(doc),contact_sheets=contacts,text_outside_page=outliers))
    json_write(DEST/'visual_geometry.json',dict(documents=records,scope='Page geometry supplements human PNG review; it does not substitute for it.'))

if __name__=='__main__':render()
