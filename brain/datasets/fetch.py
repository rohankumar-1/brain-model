"""Download the open external datasets into the dataset pool (resumable; re-running skips finished files).

    python -m brain.datasets.fetch ixi abide1 isles22

UCSF-PDGM (TCIA Aspera packages), BraTS 2023 Mets (Synapse) and ATLAS v2 (ICPSR) need a manual sign-in.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from brain.prep.sources import DATASETS, SCRATCH

IXI = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI"
S3 = "https://fcp-indi.s3.amazonaws.com"
ABIDE = "data/Projects/ABIDE_Initiative"


def curl(url: str, out: Path) -> None:
    """Resumable download; a finished file is left alone (written to ``.part`` first)."""
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    subprocess.run(["curl", "-fsSL", "-C", "-", "--retry", "5", "--retry-delay", "5", "-o", str(part), url], check=True)
    part.rename(out)


def ixi() -> None:
    for name in ("IXI-T1.tar", "IXI-T2.tar", "IXI.xls"):
        curl(f"{IXI}/{name}", DATASETS / "ixi" / name)


def isles22() -> None:
    curl("https://zenodo.org/records/7153326/files/ISLES-2022.zip", DATASETS / "isles22" / "ISLES-2022.zip")


def atlas3() -> None:
    """ATLAS R3.0 raw T1 + lesion masks, still AES-encrypted: decrypt with the emailed code (see docs)."""
    curl(f"{S3}/data/Projects/INDI/ATLAS/R3.0/atlas3_training_raw.tar.gz", DATASETS / "atlas" / "atlas3_training_raw.tar.gz")


def brats_ped() -> None:
    """BraTS 2023 pediatric glioma (train + validation) from Synapse; needs ``~/.synapseConfig`` and a joined challenge."""
    import synapseclient
    import zipfile

    syn = synapseclient.login(silent=True)
    out = DATASETS / "brats" / "2023_PED"
    for sid in ("syn51615054", "syn51929861"):
        entity = syn.get(sid, downloadLocation=str(out))
        target = DATASETS / "brats" / "extracted" / "2023_PED"
        if not (target / (Path(entity.path).stem)).exists():
            with zipfile.ZipFile(entity.path) as z:
                z.extractall(target)


def _s3_keys(prefix: str):
    token = ""
    while True:
        q = f"{S3}/?list-type=2&prefix={prefix}" + (f"&continuation-token={quote(token, safe='')}" if token else "")
        root = ET.fromstring(urlopen(q, timeout=60).read())
        ns = {"s": root.tag.split("}")[0][1:]}
        yield from (k.text for k in root.iterfind("s:Contents/s:Key", ns))
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            return
        token = root.findtext("s:NextContinuationToken", namespaces=ns)


def abide1() -> None:
    """Raw T1 (mprage) of every ABIDE I subject plus the phenotypic table (labels, age, site)."""
    out = DATASETS / "abide1"
    curl(f"{S3}/{ABIDE}/Phenotypic_V1_0b.csv", out / "Phenotypic_V1_0b.csv")
    keys = [k for k in _s3_keys(f"{ABIDE}/RawData/") if k.endswith("session_1/anat_1/mprage.nii.gz")]
    print(f"[fetch] abide1: {len(keys)} T1 volumes", flush=True)
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda k: curl(f"{S3}/{k}", out / "RawData" / k.split("RawData/")[1]), keys))


def pget(url: str, out: Path, chunk_mb: int = 512, workers: int = 16) -> None:
    """Parallel ranged download for hosts that throttle a single stream (Zenodo: 0.4 MB/s vs ~1.5 MB/s x many).
    Writes ``out.001``, ``out.002``, ... of ``chunk_mb`` MB each; 7z opens ``out.001`` as one split archive.
    Finished chunks (exact size) are skipped, so re-running resumes."""
    import re
    head = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True, check=True).stdout
    size = int(re.findall(r"(?i)content-length:\s*(\d+)", head)[-1])
    step = chunk_mb * 1024 * 1024
    n = -(-size // step)
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(i: int) -> None:
        a, b = i * step, min((i + 1) * step, size) - 1
        part = out.with_name(f"{out.name}.{i + 1:03d}")
        if part.exists() and part.stat().st_size == b - a + 1:
            return
        tmp = part.with_name(part.name + ".part")
        subprocess.run(["curl", "-fsSL", "--retry", "8", "--retry-delay", "5", "-r", f"{a}-{b}", "-o", str(tmp), url], check=True)
        tmp.rename(part)

    print(f"[fetch] {out.name}: {size / 1e9:.1f} GB in {n} chunks", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(one, range(n)), 1):
            if k % 10 == 0:
                print(f"[fetch] {k}/{n} chunks", flush=True)


def instance() -> None:
    """INSTANCE 2022 training zip (OneDrive) into the pool. Preprocess with ``python -m brain.datasets.download_instance --prep``."""
    from brain.datasets.download_instance import fetch
    fetch()




def brainmetshare() -> None:
    """BrainMetShare MRI (Stanford AIMI Redivis) into its own pool folder. See ``python -m brain.datasets.download_brainmetshare``."""
    from brain.datasets.download_brainmetshare import fetch
    fetch()


def fomo() -> None:
    """FOMO260K structural MRI (BraTS-GEN + HBN/Yale mets/NKI/CoRR/...). Scratch; mapping holdouts. See ``python -m brain.datasets.download_fomo``."""
    from brain.datasets.download_fomo import fetch
    fetch()


def aisd() -> None:
    """AISD image+mask zips (Google Drive) into the pool. Preprocess with ``python -m brain.datasets.download_aisd --prep``."""
    from brain.datasets.download_aisd import fetch
    fetch()


def isles24() -> None:
    """ISLES'24 training set (149 stroke cases: NCCT, CTA, CTP, MRI follow-up, masks), CC BY-NC-SA, 99 GB .7z on Zenodo."""
    archive = SCRATCH / "work" / "isles24" / "train.7z"
    dest = DATASETS / "isles24"
    pget("https://zenodo.org/records/17652035/files/train.7z?download=1", archive)
    if (dest / "train" / "raw_data").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    # Solid archive: include filters still decompress everything (~90 min). Keep NCCT, NCCT-space CTA, masks, phenotype xlsx.
    subprocess.run(
        ["7z", "x", str(archive.with_name(archive.name + ".001")), f"-o{dest}", "-y",
         "-ir!*_ses-01_ncct.nii.gz", "-ir!*space-ncct_cta.nii.gz",
         "-ir!*lvo-msk.nii.gz", "-ir!*cow-msk.nii.gz", "-ir!*lesion-msk.nii.gz", "-ir!*.xlsx"],
        check=True,
    )


OPENNEURO = "https://s3.amazonaws.com/openneuro.org"


def _openneuro_keys(prefix: str):
    token = ""
    while True:
        q = f"{OPENNEURO}/?list-type=2&prefix={prefix}" + (f"&continuation-token={quote(token, safe='')}" if token else "")
        root = ET.fromstring(urlopen(q, timeout=60).read())
        ns = {"s": root.tag.split("}")[0][1:]}
        yield from (k.text for k in root.iterfind("s:Contents/s:Key", ns))
        if root.findtext("s:IsTruncated", namespaces=ns) != "true":
            return
        token = root.findtext("s:NextContinuationToken", namespaces=ns)


def openneuro(ds: str, out: Path) -> None:
    """Anatomical scans of a BIDS dataset from OpenNeuro, plus the participant tables (sidecars are small)."""
    keep = []
    for k in _openneuro_keys(f"{ds}/"):
        rel = k.split("/", 1)[1]
        top = "/" not in rel and (rel.startswith("participants") or rel == "dataset_description.json")
        if top or ("/anat/" in rel and rel.startswith("sub-") and rel.endswith((".nii.gz", ".json"))):
            keep.append(k)
    print(f"[fetch] {ds}: {len(keep)} files", flush=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda k: curl(f"{OPENNEURO}/{k}", out / k.split("/", 1)[1]), keep))


def pixar() -> None:
    """Pixar short-film MRI, children 3-12 y and adults (OpenNeuro ds000228, PDDL)."""
    openneuro("ds000228", DATASETS / "pixar")


def piop1() -> None:
    """AOMIC-PIOP1, healthy young adults (OpenNeuro ds002785, CC0)."""
    openneuro("ds002785", DATASETS / "piop1")


def piop2() -> None:
    """AOMIC-PIOP2, healthy young adults (OpenNeuro ds002790, CC0)."""
    openneuro("ds002790", DATASETS / "piop2")


def id1000() -> None:
    """AOMIC-ID1000, 928 healthy adults (OpenNeuro ds003097, CC0)."""
    openneuro("ds003097", DATASETS / "id1000")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print(f"[fetch] {name}", flush=True)
        globals()[name]()
    print("[fetch] done", flush=True)
