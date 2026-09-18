"""Package checked code and deliverables; omit weights and download caches."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent


def main():
    names=['finetuning_tutorial.ipynb','finetuning_tutorial.pdf','sft_pipeline.py',
           'export_deliverables.py','validate_hw5.py','package_submission.py','requirements.txt',
           'source_revisions.json','hw5_sample_eval.csv','kaggle_test.csv','submission.csv',
           'hw5_paper_answers.md','hw5_paper_answers.pdf','hw5_writeup.pdf','README.md']
    files=[HERE/name for name in names]+sorted((HERE/'results').glob('*'))
    assert all(p.is_file() for p in files)
    archive=HERE/'hw5_submission.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path,path.relative_to(HERE).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for path in files:
            assert z.read(path.relative_to(HERE).as_posix())==path.read_bytes()
    print(f'Created {archive.name}: {len(files)} files, {archive.stat().st_size:,} bytes')


if __name__=='__main__':main()
