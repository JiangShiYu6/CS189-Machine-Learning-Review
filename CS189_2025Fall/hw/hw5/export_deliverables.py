"""Export the written answers, full executed notebook and two-page report."""
from pathlib import Path
import base64
import io
import json
import re
import textwrap
from html.parser import HTMLParser
from xml.sax.saxutils import escape
import nbformat
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Preformatted,
                               Image, PageBreak, Table, TableStyle, KeepTogether)

HERE = Path(__file__).resolve().parent
STYLES = getSampleStyleSheet()
for name in ['Heading1','Heading2','Heading3']:
    STYLES[name].keepWithNext=True
STYLES.add(ParagraphStyle('BodySmall', parent=STYLES['BodyText'], fontSize=9, leading=12, spaceAfter=6))
STYLES.add(ParagraphStyle('CodeSmall', fontName='Courier', fontSize=7, leading=9))
for candidate in ['C:/Windows/Fonts/consola.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf']:
    if Path(candidate).exists():
        pdfmetrics.registerFont(TTFont('CodeUnicode',candidate))
        STYLES['CodeSmall'].fontName='CodeUnicode'
        break


def inline(text):
    text = escape(text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#17578a">\1</link>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
    return re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)


def markdown_blocks(text, small=False):
    result=[]
    text=re.sub(r'(?m)^(#{1,6} [^\n]+)\n(?=\S)',r'\1\n\n',text)
    for block in re.split(r'\n\s*\n', text.strip()):
        if block.startswith('#'):
            depth=len(block)-len(block.lstrip('#'))
            style=STYLES['Title' if depth==1 else 'Heading2' if depth==2 else 'Heading3']
            result.append(Paragraph(inline(block.lstrip('# ')), style))
        else:
            result.append(Paragraph(inline(block).replace('\n','<br/>'), STYLES['BodySmall' if small else 'BodyText']))
        if not block.startswith('#'):
            result.append(Spacer(1,4))
    return result


def footer(canvas, doc):
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(42,25,'CS189 HW5 | Reproducible SFT experiment')
    canvas.drawRightString(570,25,str(doc.page))


def build_pdf(name, story, **kwargs):
    doc=SimpleDocTemplate(str(HERE/name), pagesize=(612,792), rightMargin=42,leftMargin=42,
                          topMargin=35,bottomMargin=38,**kwargs)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def code_lines(text, width=112):
    # Preserve every code/output character, wrapping long lines instead of clipping.
    lines=[]
    for line in text.expandtabs(4).splitlines():
        lines.extend(textwrap.wrap(line, width=width, replace_whitespace=False,
                                   drop_whitespace=False, break_long_words=True,
                                   break_on_hyphens=False) or [''])
    return '\n'.join(lines)


class VisibleHTML(HTMLParser):
    """Preserve visible notebook HTML output, including Trainer's loss table."""
    def __init__(self):
        super().__init__(); self.parts=[]; self.hidden=False
    def handle_starttag(self,tag,attrs):
        if tag in ('style','script'):self.hidden=True
    def handle_endtag(self,tag):
        if tag in ('style','script'):self.hidden=False
        if tag in ('td','th'):self.parts.append(' | ')
        if tag in ('tr','p','div','br'):self.parts.append('\n')
    def handle_data(self,data):
        if not self.hidden:
            self.parts.append(re.sub(r'\s+',' ',data))
    def text(self):
        return '\n'.join(line.strip() for line in ''.join(self.parts).splitlines() if line.strip())


def export_notebook():
    nb=nbformat.read(HERE/'finetuning_tutorial.ipynb',as_version=4)
    story=[Paragraph('HW5: executed notebook',STYLES['Title'])]
    for index,cell in enumerate(nb.cells):
        if cell.cell_type=='markdown':
            story.extend(markdown_blocks(cell.source,small=True))
        elif cell.cell_type=='code':
            story.append(Paragraph(f'Code cell {index} | execution {cell.execution_count}',STYLES['Heading3']))
            story.append(Preformatted(code_lines(cell.source),STYLES['CodeSmall']))
            for output in cell.get('outputs',[]):
                if output.output_type=='error':
                    raise ValueError(f'Notebook contains an error in cell {index}')
                data=output.get('data',{})
                if 'image/png' in data:
                    pic=Image(io.BytesIO(base64.b64decode(data['image/png'])))
                    ratio=min(528/pic.imageWidth,300/pic.imageHeight)
                    pic.drawWidth=pic.imageWidth*ratio;pic.drawHeight=pic.imageHeight*ratio
                    story.append(pic)
                elif 'text/html' in data:
                    parser=VisibleHTML();parser.feed(data['text/html'])
                    story.append(Preformatted(code_lines(parser.text()),STYLES['CodeSmall']))
                elif output.output_type=='stream' or 'text/plain' in data:
                    value=output.get('text',data.get('text/plain',''))
                    value=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',value).replace('\r','\n')
                    story.append(Preformatted(code_lines(value),STYLES['CodeSmall']))
            story.append(Spacer(1,10))
    story.append(PageBreak())
    story.append(Paragraph('Appendix: artifact and validation scripts',STYLES['Title']))
    for filename in ['export_deliverables.py','validate_hw5.py','package_submission.py']:
        story.append(Paragraph(filename,STYLES['Heading2']))
        lines=code_lines((HERE/filename).read_text(encoding='utf-8')).splitlines()
        for start in range(0,len(lines),40):
            story.append(Preformatted('\n'.join(lines[start:start+40]),STYLES['CodeSmall']))
    build_pdf('finetuning_tutorial.pdf',story)


def export_report():
    results=HERE/'results'
    metrics=json.loads((results/'metrics.json').read_text())
    manifest=json.loads((results/'manifest.json').read_text())
    examples=json.loads((results/'examples.json').read_text())
    story=[Paragraph('HW5 SFT experiment report',STYLES['Title'])]
    method=(f'I fine-tuned Qwen2.5-0.5B-Instruct with LoRA on {manifest["train_count"]} four-choice '
      'MMLU questions from machine learning, computer science, college mathematics, statistics, history, '
      'geography, nutrition, and elementary mathematics, using at most 120 training questions per subject [1,2]. '
      'As in the starter, MMLU test splits supply training data; dev/validation splits are held out, and questions '
      'matching either course CSV are removed before training. '
      'The final run uses two epochs, LoRA rank 16/alpha 32/dropout 0.05 on all linear layers, AdamW at peak '
      'learning rate 1e-4, weight decay 0.01, cosine decay, 10% warmup, batch 1 with 16-step accumulation, '
      'gradient clipping 1, bfloat16 and seed 189 [3]. '
      'SFTTrainer minimizes completion-only cross-entropy on boxed answer letters after deterministic choice '
      'permutation; base and tuned models use the same constrained letter decoding [4]. '
      'Mixing general and specialized examples is motivated by data-composition work [5], but this one-stage '
      'experiment does not reproduce DMT.')
    story.extend(markdown_blocks('## 1. Final method\n\n'+method,True))
    rows=[['Model','CS189 public (25)','Specialized holdout','General holdout']]
    for name in ['base','narrow','mixed']:
        row=[name]
        for group in ['cs189','specialized','general']:
            v=metrics[name][group];row.append(f'{v["correct"]}/{v["n"]} ({v["accuracy"]:.1%})')
        rows.append(row)
    table=Table(rows,colWidths=[65,145,159,159])
    table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),
                               ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),
                               ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(table)
    story.append(Image(str(results/'loss_curves.png'),width=440,height=189))
    narrow=metrics['narrow'];base=metrics['base']
    # Describe measured changes, without assuming an attempted method necessarily failed.
    delta=narrow['specialized']['accuracy']-base['specialized']['accuracy']
    experiment=(f'The narrow attempt repeatedly trained only machine-learning questions for 30 optimizer steps '
      f'at learning rate 3e-4; its specialized held-out accuracy changed by {delta*100:+.1f} percentage points, '
      f'while general accuracy changed from {base["general"]["accuracy"]:.1%} to {narrow["general"]["accuracy"]:.1%}. '
      'Its limitations are consistent with fitting a small, narrow dataset and shifting answer preferences; '
      'this comparison changes both data coverage and training budget, so it cannot isolate their causal effects.')
    story.extend(markdown_blocks('## 2. Narrow-data attempt\n\n'+experiment,True))
    story.append(Paragraph('Public accuracy is descriptive: the 25-question public set overlaps the hidden CSV. '
                           'No public or hidden questions were trained on, and hidden accuracy is unavailable. '
                           'The held-out MMLU results are local split evaluations, not official benchmark scores.',STYLES['BodySmall']))
    story.append(PageBreak())
    story.append(Paragraph('3-4. Real changes on non-CS189 inputs',STYLES['Heading2']))
    story.append(Paragraph('Each input uses the same instruction: choose one option and return its letter in a LaTeX box. '
                           'The examples below are the two shortest observed changes of each type in general held-out subjects. '
                           'These illustrate individual forgetting and gains, rather than proving a broad loss or gain of an entire skill.',STYLES['BodySmall']))
    for category in ['forgetting','gains']:
        for i,ex in enumerate(examples[category],1):
            label='Forgetting' if category=='forgetting' else 'Gain'
            story.append(Paragraph(f'{label} {i} | {escape(ex["source"])}',STYLES['Heading3']))
            content=escape(ex['question'])+'<br/>'+' &nbsp;&nbsp; '.join(
                f'{chr(65+j)}. {escape(choice)}' for j,choice in enumerate(ex['choices']))
            story.append(Paragraph(content,STYLES['BodySmall']))
            story.append(Paragraph(escape(f'Base: {ex["base_output"]}; final: {ex["final_output"]}; correct: {ex["answer"]}.'),STYLES['BodySmall']))
            explanations={
                'forgetting':[
                    'The tuned model loses a previously correct linear-equation answer; interference from the limited answer-only training mixture is one possible cause.',
                    'The tuned model chooses the negative of the correct solution, suggesting that the update changed its handling of signs on this input.'],
                'gains':[
                    'Exposure to other elementary-mathematics examples may have improved percentage reasoning on this held-out question.',
                    'The related arithmetic training may have improved this signed-sum answer, although the output alone cannot establish the mechanism.']}
            explanation=explanations[category][i-1]
            story.append(Paragraph(explanation,STYLES['BodySmall']))
    story.append(Paragraph('Kaggle status',STYLES['Heading3']))
    story.append(Paragraph('The 169-row submission.csv is generated and checked. Kaggle username and score screenshot '
                           'are pending authenticated submission; no leaderboard score is claimed.',STYLES['BodySmall']))
    story.append(Paragraph('Sources and acknowledgement',STYLES['Heading3']))
    refs='[1] <link href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct">Qwen model card</link>. '
    refs+='[2] <link href="https://huggingface.co/datasets/cais/mmlu">cais/mmlu</link>. '
    refs+='[3] <link href="https://arxiv.org/abs/2106.09685">Hu et al., LoRA</link>. '
    refs+='[4] <link href="https://huggingface.co/docs/trl/v0.24.0/sft_trainer">TRL SFTTrainer</link>. '
    refs+='[5] <link href="https://arxiv.org/abs/2310.05492">Dong et al., SFT data composition</link>. '
    refs+='OpenAI Codex assisted with code, debugging and writing; results come from actual model runs.'
    story.append(Paragraph(refs,STYLES['BodySmall']))
    build_pdf('hw5_writeup.pdf',story)


if __name__=='__main__':
    import sys
    build_pdf('hw5_paper_answers.pdf',markdown_blocks((HERE/'hw5_paper_answers.md').read_text(encoding='utf-8')))
    if '--paper-only' not in sys.argv:
        export_notebook()
        export_report()
    print('PDF exports complete')
