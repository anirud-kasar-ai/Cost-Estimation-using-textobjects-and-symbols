"""One-time prep: fixed data.yaml files + remap small test labels 8c -> 9c."""
from pathlib import Path
import shutil

work = Path(r"D:\Cost Estimation Using Text and Object\eval_data")

# ---- Main project: yaml with absolute paths ----
tp = work / "test_project"
names52 = ['APWP', 'CONDUIT STUB', 'CONDUIT STUB 2', 'DATA DROP LOCATION  INTERIORSPEAKER', 'DATA DROP LOCATION SPEAKER', 'DATA PERMANENT LINK', 'DATA POLE', 'DATA RACK', 'EXHAUST FAN', 'EXTERIOR WIRELESS ACESS POINT HIGH DENSITY', 'FIRE ALARAM CONTROL PANNEL', 'GROUND BOX', 'GROUND BUS BAR', 'HEAT STACK', 'IACP', 'INTERCOM CONSOLE', 'INTERIOR ENCLOSURE', 'INTERIOR SPEAKER', 'INTERIOR WIRELESS ACESS POINT', 'INTERIOR WIRELESS ACESS POINT HIGH DENSITY', 'IP CONSOLE PHONE', 'J HOOK', 'JUNCTION BOX', 'LADDER', 'LAY IN SPEAKERIP MODULE', 'LAYER INTERCOM SPEAKER 25V', 'MINIMUM POINT OF ENTRY', 'MODULE BOGEN', 'MODULE COMBO BOX', 'MOUNT', 'NA-LO', 'NEMA4', 'NETWORK CAMERA', 'NETWORK VIDEO RECORDER', 'NQ-CC', 'NQ-MIX', 'PHONE HANDSETEQUIPMENT', 'PLUMBING STACK', 'ROOF DRAIN', 'ROOF HATCH', 'SIGNAL TERMINAL CABINATE', 'SINGEL NETWORK CAMERA', 'SURFACE RACEWAY WM2300', 'SURFACE REACEWAY WM 5400', 'SURFACE REACEWAY WM 5500', 'TEL', 'VENT INTAKE', 'WALL MOUNT', 'WINDOW MARK AND TYPE', 'WIRE GUARD', 'classroom 15', 'object']
yaml_main = tp / "data_fixed.yaml"
yaml_main.write_text(
    f"train: {(tp/'train'/'images').as_posix()}\n"
    f"val: {(tp/'valid'/'images').as_posix()}\n"
    f"test: {(tp/'test'/'images').as_posix()}\n"
    f"nc: 52\nnames: {names52}\n"
)
print("wrote", yaml_main)

# ---- Small project: remap 8-class labels -> 9-class (PIPE HOUSING inserted at idx 4) ----
src = work / "test_small"
dst = work / "test_small_9c"
remap = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7, 7: 8}
for split in ["valid", "test"]:
    (dst / split / "labels").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / split / "images", dst / split / "images", dirs_exist_ok=True)
    for lf in (src / split / "labels").glob("*.txt"):
        out = []
        for line in lf.read_text().splitlines():
            p = line.split()
            if p:
                p[0] = str(remap[int(p[0])])
                out.append(" ".join(p))
        (dst / split / "labels" / lf.name).write_text("\n".join(out) + "\n")

names9 = ['CURB', 'EXHAUST FAN', 'HATCH DETAIL', 'HEAT DETAIL', 'PIPE HOUSING', 'PIPE PENETRATION', 'PLUMBING STACK', 'ROOF DRAIN', 'SKYLIGHT']
yaml_small = dst / "data_fixed.yaml"
yaml_small.write_text(
    f"train: {(dst/'valid'/'images').as_posix()}\n"
    f"val: {(dst/'valid'/'images').as_posix()}\n"
    f"test: {(dst/'test'/'images').as_posix()}\n"
    f"nc: 9\nnames: {names9}\n"
)
print("wrote", yaml_small)

# GT instance counts per split for sanity checks later
for name, d in [("main valid", tp / "valid" / "labels"), ("main test", tp / "test" / "labels"),
                ("small valid", dst / "valid" / "labels"), ("small test", dst / "test" / "labels")]:
    n = sum(len([l for l in f.read_text().splitlines() if l.strip()]) for f in d.glob("*.txt"))
    print(f"{name}: {n} ground-truth objects")
