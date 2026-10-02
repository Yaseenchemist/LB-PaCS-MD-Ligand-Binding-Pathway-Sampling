#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, re, shutil, subprocess, sys, tarfile, tempfile, time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import yaml
import MDAnalysis as mda
from MDAnalysis.analysis import align

ROOT = Path(__file__).resolve().parents[1]

class PipelineError(RuntimeError): pass

def log(msg):
    print(time.strftime('[%Y-%m-%d %H:%M:%S]'), msg, flush=True)

def run(cmd, cwd=None, input_text=None, capture=False, env=None, check=True):
    log('$ ' + ' '.join(map(str, cmd)))
    r = subprocess.run([str(x) for x in cmd], cwd=cwd, input=input_text, text=True,
                       stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.STDOUT if capture else None,
                       env=env, check=False)
    if check and r.returncode != 0:
        out = r.stdout or ''
        raise PipelineError(f"Command failed ({r.returncode}): {' '.join(map(str,cmd))}\n{out[-6000:]}")
    return r

def ensure(cond, msg):
    if not cond: raise PipelineError(msg)

def load_cfg(path):
    with open(path) as f: return yaml.safe_load(f)

def mkdir(p): Path(p).mkdir(parents=True, exist_ok=True)

def done(path): Path(path).write_text(time.strftime('%F %T')+'\n')

def is_done(path): return Path(path).exists()

def require_exe(name):
    p=shutil.which(name)
    ensure(p, f"Required executable not found: {name}")
    return p

def read_text(p): return Path(p).read_text(errors='replace')
def write_json(p,obj): Path(p).write_text(json.dumps(obj,indent=2))

def setup_gmxlib(cfg):
    ffpat=cfg['forcefield']['preferred_pattern']
    found=sorted(ROOT.glob(ffpat))
    if not os.environ.get('GMXLIB'):
        ensure(found, f"No force-field directory matching {ffpat} beside project and GMXLIB not set")
        os.environ['GMXLIB']=str(ROOT)
    if cfg['forcefield'].get('gmx_ff_name'):
        ffname=cfg['forcefield']['gmx_ff_name']
    elif found:
        ffname=found[0].name[:-3] if found[0].name.endswith('.ff') else found[0].name
    else:
        # If GMXLIB is externally set, discover there.
        g=Path(os.environ['GMXLIB'])
        f=sorted(g.glob(ffpat)); ensure(f, f"Could not infer force field in GMXLIB={g}")
        ffname=f[0].name[:-3]
    return ffname

def preflight(cfg, profile):
    require_exe(cfg['execution']['gmx']); require_exe('obabel'); require_exe('curl')
    if profile=='slurm_gpu':
        require_exe('sbatch'); require_exe('squeue')
    try:
        import pexpect  # noqa
    except Exception as e: raise PipelineError('Python package pexpect is required') from e
    gmx=cfg['execution']['gmx']
    r=run([gmx,'--version'],capture=True)
    text=r.stdout or ''
    log(text.splitlines()[0] if text else 'GROMACS detected')
    if profile=='slurm_gpu' and 'CUDA' not in text.upper():
        log('WARNING: current front-node GROMACS version text does not explicitly show CUDA; GPU job will verify at runtime.')
    ff=setup_gmxlib(cfg)
    log(f"GMXLIB={os.environ.get('GMXLIB')} ; force field={ff} ; profile={profile}")
    return ff

# ---------- ligand/protein detection ----------
WATER_DEFAULT={'SOL','HOH','WAT','TIP3','TIP3P'}
ION_DEFAULT={'NA','CL','K','CA','MG','ZN','MN','FE'}

def guess_ligand(u, override=None):
    if override:
        ag=u.select_atoms(f"resname {override}")
        ensure(len(ag)>0, f"ligand_resname_override={override} not found")
        return override
    # Strong preference for common docking ligand names.
    names=[r.resname.strip() for r in u.residues]
    for pref in ('LIG','UNL','MOL'):
        if pref in names: return pref
    protein_residx=set(u.select_atoms('protein').resindices.tolist())
    candidates=[]
    for r in u.residues:
        rn=r.resname.strip()
        if r.resindex in protein_residx or rn in WATER_DEFAULT or rn in ION_DEFAULT: continue
        heavy=sum(1 for a in r.atoms if (getattr(a,'element','') or a.name[:1]).upper()!='H')
        carbon=sum(1 for a in r.atoms if ((getattr(a,'element','') or a.name[:1]).upper()=='C'))
        if heavy>=5 and carbon>=2: candidates.append((heavy,len(r.atoms),rn,r.resid,r.resindex))
    ensure(candidates, 'Could not automatically identify an organic non-protein ligand. Set ligand_resname_override.')
    candidates.sort(reverse=True)
    # Reject near-ties with different resnames rather than silently choosing a cofactor.
    if len(candidates)>1 and candidates[1][0] >= 0.9*candidates[0][0] and candidates[1][2]!=candidates[0][2]:
        raise PipelineError(f"Ambiguous ligand candidates: {candidates[:4]}. Set ligand_resname_override in config.yaml")
    return candidates[0][2]

def stage_detect(cfg,w):
    st=w/'01_detection'; mkdir(st)
    if is_done(st/'.done'): return json.loads((st/'system.json').read_text())
    src=ROOT/cfg['project']['input_complex']; ensure(src.exists(),f'Missing {src}')
    u=mda.Universe(str(src))
    ligres=guess_ligand(u,cfg['project'].get('ligand_resname_override'))
    lig=u.select_atoms(f"resname {ligres}")
    ensure(len(lig)>0,'Ligand selection is empty')
    protein=u.select_atoms('protein')
    ensure(len(protein)>0,'MDAnalysis did not detect protein atoms')
    # Pocket fixed from original docking pose.
    pocket=u.select_atoms(f"protein and byres (around {cfg['project']['pocket_cutoff_A']} resname {ligres})")
    ensure(len(pocket.residues)>0,'No pocket residues found around ligand')
    protein.write(str(st/'protein.pdb')); lig.write(str(st/'ligand_pose.pdb'))
    pockets=[{'resid':int(r.resid),'resname':r.resname.strip(),'segid':str(r.segid).strip()} for r in pocket.residues]
    meta={'ligand_resname':ligres,'ligand_atoms':len(lig),'protein_atoms':len(protein),'pocket_residues':pockets}
    write_json(st/'system.json',meta); done(st/'.done'); log(f"Detected ligand {ligres}; pocket residues={len(pockets)}")
    return meta

# ---------- SwissParam / topology parsing ----------
def swissparam(cfg,mol2,outdir):
    mkdir(outdir); sp=cfg['swissparam']; host=sp['host'].rstrip('/')
    query=f"approach={sp['approach']}"
    if sp.get('legacy_flag'): query += '&'+sp['legacy_flag'].lstrip('&')
    # Health check first.
    for attempt in range(sp['retries']):
        r=run(['curl','--fail','--silent','--show-error','--max-time','30',host+'/'],capture=True,check=False)
        if r.returncode==0: break
        time.sleep(5*(attempt+1))
    else: raise PipelineError('SwissParam server health check failed')
    r=run(['curl','--fail','--silent','--show-error','-F',f'myMol2=@{mol2}',f'{host}/startparam?{query}'],capture=True)
    txt=r.stdout or ''; m=re.search(r'(?:Session number\s*:\s*)?(\d{5,})',txt,re.I)
    ensure(m,f'Could not parse SwissParam session number from: {txt[:500]}')
    sid=m.group(1); (outdir/'session.txt').write_text(sid+'\n'); log(f'SwissParam session={sid}')
    deadline=time.time()+sp['timeout_minutes']*60
    while time.time()<deadline:
        r=run(['curl','--fail','--silent','--show-error',f'{host}/checksession?sessionNumber={sid}'],capture=True,check=False)
        txt=r.stdout or ''; log('SwissParam: '+txt.strip()[:300])
        if r.returncode==0 and re.search(r'finished|complete|successful',txt,re.I): break
        if re.search(r'failed|error',txt,re.I): raise PipelineError('SwissParam reported failure: '+txt)
        time.sleep(sp['poll_seconds'])
    else: raise PipelineError('SwissParam timeout')
    tgz=outdir/'results.tar.gz'
    run(['curl','--fail','--silent','--show-error',f'{host}/retrievesession?sessionNumber={sid}','-o',str(tgz)])
    ensure(tarfile.is_tarfile(tgz),'SwissParam retrieve result is not a valid tar archive')
    with tarfile.open(tgz) as t: t.extractall(outdir/'raw')
    itps=[p for p in (outdir/'raw').rglob('*.itp') if 'posre' not in p.name.lower()]
    ensure(itps,'SwissParam archive contains no ligand .itp')
    return itps[0], next(iter(list((outdir/'raw').rglob('*.prm'))+list((outdir/'raw').rglob('*.par'))),None)

def split_itp(src,atomtypes_out,mol_out):
    lines=read_text(src).splitlines(True)
    section=None; global_secs={'atomtypes','pairtypes','nonbond_params','bondtypes','angletypes','dihedraltypes','constrainttypes'}
    g=[]; m=[]
    for line in lines:
        mm=re.match(r'^\s*\[\s*([^\]]+)\s*\]',line)
        if mm: section=mm.group(1).strip().lower()
        (g if section in global_secs else m).append(line)
    # Remove leading comments/blanks accidentally duplicated in mol only harmless.
    Path(atomtypes_out).write_text(''.join(g)); Path(mol_out).write_text(''.join(m))
    mt=None; inmt=False
    for line in m:
        mm=re.match(r'^\s*\[\s*moleculetype\s*\]',line,re.I)
        if mm: inmt=True; continue
        if inmt and line.strip() and not line.lstrip().startswith(';'):
            mt=line.split()[0]; break
    ensure(mt,'Cannot parse [ moleculetype ] from SwissParam ITP')
    return mt

def parse_itp_atoms(path):
    atoms=[]; sec=None
    for line in read_text(path).splitlines():
        m=re.match(r'^\s*\[\s*([^\]]+)\s*\]',line)
        if m: sec=m.group(1).strip().lower(); continue
        if sec=='atoms' and line.strip() and not line.lstrip().startswith(';'):
            x=line.split(';',1)[0].split()
            if len(x)>=5:
                atoms.append({'nr':int(x[0]),'type':x[1],'resnr':x[2],'residue':x[3],'atom':x[4], 'mass': float(x[7]) if len(x)>7 else None})
    ensure(atoms,'No [ atoms ] parsed from ligand topology')
    return atoms

def make_lig_posres(itp,out,fc):
    atoms=parse_itp_atoms(itp); rows=[]
    for a in atoms:
        heavy=(a['mass'] is None or a['mass']>2.0) and not a['atom'].upper().startswith('H')
        if heavy: rows.append(f"{a['nr']:6d}  1  {fc:8.1f} {fc:8.1f} {fc:8.1f}\n")
    ensure(rows,'No ligand heavy atoms found for restraints')
    Path(out).write_text('[ position_restraints ]\n; ai funct fcx fcy fcz\n'+''.join(rows))

def stage_param(cfg,w,meta):
    st=w/'00_ligand'; mkdir(st)
    if is_done(st/'.done'): return json.loads((st/'ligand.json').read_text())
    pose=w/'01_detection'/'ligand_pose.pdb'
    # One unminimized H-added copy preserves heavy-atom docking geometry for MD coordinates.
    run(['obabel',str(pose),'-O',str(st/'ligand_pose_H.mol2'),'-h'])
    # Separate minimized copy is used ONLY for parameter generation.
    run(['obabel',str(st/'ligand_pose_H.mol2'),'-O',str(st/'ligand_param.mol2'),'--minimize','--ff','MMFF94','--steps','1000'])
    itp,par=swissparam(cfg,st/'ligand_param.mol2',st/'swissparam')
    mt=split_itp(itp,st/'ligand_atomtypes.itp',st/'ligand_mol.itp')

    # SwissParam may use UNL1 as both moleculetype and residue name.
    # Keep the SwissParam moleculetype unchanged, but make the residue
    # name in [ atoms ] consistent with the original complex (e.g. LIG).
    mol_itp = st/'ligand_mol.itp'
    lines = mol_itp.read_text().splitlines()
    patched = []
    in_atoms = False

    for line in lines:
        s = line.strip()

        if s.startswith('[') and s.endswith(']'):
            section = s.strip('[]').strip().lower()
            in_atoms = (section == 'atoms')
            patched.append(line)
            continue

        if in_atoms and s and not s.startswith(';'):
            body, sep, comment = line.partition(';')
            fields = body.split()

            # GROMACS [ atoms ]:
            # nr type resnr residue atom cgnr charge [mass]
            if len(fields) >= 7:
                fields[3] = str(meta['ligand_resname']).strip()
                line = ' '.join(fields)
                if sep:
                    line += ' ;' + comment

        patched.append(line)

    mol_itp.write_text('\n'.join(patched) + '\n')

    if par: shutil.copy2(par,st/'ligand_extra.prm')
    atoms=parse_itp_atoms(st/'ligand_mol.itp')
    # Convert unminimized hydrogenated pose and verify count before renaming by topology order.
    run(['obabel',str(st/'ligand_pose_H.mol2'),'-O',str(st/'ligand_pose_H.pdb')])
    u=mda.Universe(str(st/'ligand_pose_H.pdb'))
    ensure(len(u.atoms)==len(atoms),f"Ligand atom-count mismatch: coordinate={len(u.atoms)} topology={len(atoms)}")
    for a,ta in zip(u.atoms,atoms): a.name=ta['atom']
    # Keep original ligand resname for coordinate selections; topology moleculetype can differ.
    for r in u.residues: r.resname=meta['ligand_resname']
    u.atoms.write(str(st/'ligand_ready.pdb'))
    make_lig_posres(st/'ligand_mol.itp',st/'posre_ligand.itp',cfg['system']['ligand_posres_fc'])
    info={'moleculetype':mt,'atom_count':len(atoms),'has_extra_prm':bool(par)}
    write_json(st/'ligand.json',info); done(st/'.done'); return info

# ---------- pdb2gmx interactive termini ----------
def pdb2gmx_termini(gmx, protein, outgro, top, ff, water, cfg, cwd):
    import pexpect

    gmx_exe = shutil.which(gmx)
    ensure(
        gmx_exe is not None,
        f"GROMACS executable not found in controller environment: {gmx}"
    )

    args = [
        'pdb2gmx',
        '-f', str(protein),
        '-o', str(outgro),
        '-p', str(top),
        '-ignh',
        '-ff', str(ff),
        '-water', str(water),
        '-ter',
    ]

    child = pexpect.spawn(
        gmx_exe,
        args,
        cwd=str(cwd),
        encoding='utf-8',
        timeout=180,
    )

    transcript = []

    nre = re.compile(
        cfg['forcefield']['termini']['nterm_regex'],
        re.I
    )
    cre = re.compile(
        cfg['forcefield']['termini']['cterm_regex'],
        re.I
    )

    option_pattern = re.compile(
        r'(?m)^\s*(\d+)\s*:\s*([^\r\n]+)\r?$'
    )

    def choose_terminus(rx, which):
        """
        Read numbered GROMACS terminus entries directly from the
        pexpect buffer until a label matches the configured regex.
        """
        seen = []

        while True:
            idx = child.expect(
                [
                    option_pattern,
                    pexpect.EOF,
                    pexpect.TIMEOUT,
                ],
                timeout=3
            )

            transcript.append(child.before or '')

            if idx == 0:
                number = int(child.match.group(1))
                label = child.match.group(2).strip()

                transcript.append(
                    f"{number}: {label}\n"
                )
                seen.append((number, label))

                if rx.search(label):
                    transcript.append(
                        f"AUTO_CHOICE {which} {number}: {label}\n"
                    )
                    child.sendline(str(number))
                    return

            elif idx == 1:
                raise PipelineError(
                    f"pdb2gmx ended while selecting {which} terminus. "
                    f"Options seen: {seen}"
                )

            else:
                raise PipelineError(
                    f"Could not auto-select {which} terminus "
                    f"using regex {rx.pattern}. "
                    f"Options seen: {seen}"
                )

    while True:
        idx = child.expect(
            [
                r'Select start terminus type[^\r\n]*',
                r'Select end terminus type[^\r\n]*',
                pexpect.EOF,
                pexpect.TIMEOUT,
            ],
            timeout=180
        )

        transcript.append(child.before or '')

        if idx == 0:
            transcript.append((child.after or '') + '\n')
            choose_terminus(nre, 'N')

        elif idx == 1:
            transcript.append((child.after or '') + '\n')
            choose_terminus(cre, 'C')

        elif idx == 2:
            break

        else:
            (Path(cwd) / 'pdb2gmx.log').write_text(
                ''.join(transcript)
            )
            child.close(force=True)
            raise PipelineError(
                'pdb2gmx terminal selection timed out'
            )

    child.close()

    (Path(cwd) / 'pdb2gmx.log').write_text(
        ''.join(transcript)
    )

    ensure(
        child.exitstatus == 0,
        f'pdb2gmx failed with exit status {child.exitstatus}; '
        f'see {cwd}/pdb2gmx.log'
    )

def gro_merge(a,b,out):
    A=Path(a).read_text().splitlines(); B=Path(b).read_text().splitlines()
    na=int(A[1]); nb=int(B[1]);
    Path(out).write_text('\n'.join([A[0],str(na+nb),*A[2:-1],*B[2:-1],A[-1]])+'\n')

def patch_topology(top,ligdir,liginfo):
    p=Path(top); txt=p.read_text()
    inc_atom='#include "ligand_atomtypes.itp"\n'
    inc_extra=''
    inc_mol='#include "ligand_mol.itp"\n' 
    # atom types / parameter additions after forcefield.itp, before any moleculetype.
    txt=re.sub(r'(#include\s+"[^"]*forcefield\.itp"\s*\n)',lambda m:m.group(1)+'; ligand global parameters\n'+inc_atom+inc_extra,txt,count=1)
    ensure('ligand_atomtypes.itp' in txt,'Could not insert ligand atomtypes after forcefield.itp')
    txt=re.sub(r'(?m)^\s*\[\s*system\s*\]', '; ligand molecule topology\n'+inc_mol+'\n[ system ]',txt,count=1)
    ensure('ligand_mol.itp' in txt,'Could not insert ligand moleculetype before [ system ]')
    # Add ligand to [molecules] before any solvent that solvate will later append.
    txt += f"\n; ligand count\n{liginfo['moleculetype']}    1\n" if not re.search(r'(?m)^\s*'+re.escape(liginfo['moleculetype'])+r'\s+1\s*$',txt) else ''
    p.write_text(txt)

def copy_lig_includes(ligdir,sysdir):
    for n in ['ligand_atomtypes.itp','ligand_mol.itp','posre_ligand.itp','ligand_extra.prm']:
        src=Path(ligdir)/n
        if src.exists(): shutil.copy2(src,Path(sysdir)/n)

def write_index_from_coords(coord, meta, cfg, out):
    u=mda.Universe(str(coord)); lig=u.select_atoms(f"resname {meta['ligand_resname']}"); prot=u.select_atoms('protein')
    water=' '.join(cfg['system']['water_resnames']); ions=' '.join(cfg['system']['ion_resnames'])
    wai=u.select_atoms(f"resname {water} {ions}")
    pl=prot+lig
    ensure(len(lig)>0 and len(prot)>0 and len(wai)>0,'Failed to build required index groups')
    def grp(name,ag):
        ids=ag.indices+1; lines=[f'[{name}]']
        for i in range(0,len(ids),15): lines.append(' '.join(map(str,ids[i:i+15])))
        return '\n'.join(lines)+'\n'
    Path(out).write_text(grp('Protein_LIG',pl)+grp('Water_and_ions',wai)+grp('LIGAND',lig)+grp('Protein',prot))

def stage_build(cfg,w,meta,liginfo,ff):
    st=w/'02_system'; mkdir(st)
    if is_done(st/'.done'): return
    copy_lig_includes(w/'00_ligand',st)
    protein=(w/'01_detection'/'protein.pdb').resolve()
    pdb2gmx_termini(cfg['execution']['gmx'],protein,'protein.gro','topol.top',ff,cfg['forcefield']['water'],cfg,st)
    run(['obabel',str(w/'00_ligand'/'ligand_ready.pdb'),'-O',str(st/'ligand_ready.gro')])

    # OpenBabel can rename the ligand residue to UNL1 when writing GRO.
    # Restore the ligand residue name detected from the original complex so
    # all downstream MDAnalysis selections use one consistent resname.
    gro = st/'ligand_ready.gro'
    glines = gro.read_text().splitlines()
    ensure(len(glines) >= 3, 'Invalid ligand_ready.gro produced by OpenBabel')

    lig_resname = str(meta['ligand_resname']).strip()
    ensure(
        1 <= len(lig_resname) <= 5,
        f"Ligand residue name cannot be represented safely in GRO: {lig_resname}"
    )

    fixed = [glines[0], glines[1]]
    for line in glines[2:-1]:
        ensure(len(line) >= 10, 'Malformed atom line in ligand_ready.gro')
        fixed.append(line[:5] + f"{lig_resname:>5}" + line[10:])
    fixed.append(glines[-1])

    gro.write_text('\n'.join(fixed) + '\n')

    gro_merge(st/'protein.gro',st/'ligand_ready.gro',st/'complex.gro')
    patch_topology(st/'topol.top',st,liginfo)
    gmx=cfg['execution']['gmx']
    run([gmx,'editconf','-f','complex.gro','-o','box.gro','-bt',cfg['system']['box_type'],'-d',str(cfg['system']['box_distance_nm'])],cwd=st)
    run([gmx,'solvate','-cp','box.gro','-cs','spc216.gro','-p','topol.top','-o','solv.gro'],cwd=st)
    # Pre-genion system may have net charge; genion neutralizes it.
    run([gmx,'grompp','-f',str(ROOT/'mdp'/'ions.mdp'),
         '-c','solv.gro','-p','topol.top','-o','ions.tpr',
         '-maxwarn','1'],cwd=st)
    # Determine SOL group by name, not group number.
    run([gmx,'genion','-s','ions.tpr','-p','topol.top','-conc',str(cfg['system']['salt_concentration_M']),'-neutral','-o','solv_ions.gro'],cwd=st,input_text='SOL\n')
    write_index_from_coords(st/'solv_ions.gro',meta,cfg,st/'index.ndx')
    done(st/'.done')

# ---------- execution abstraction ----------
def gpu_flags():
    return ['-nb','gpu','-pme','gpu','-bonded','gpu','-update','gpu']

def local_mdrun(cfg,cwd,deffnm,is_em=False):
    g=cfg['execution']['gmx']; th=cfg['execution']['local_threads']
    return run([g,'mdrun','-v','-deffnm',deffnm,'-ntmpi','1','-ntomp',str(th)],cwd=cwd)

def slurm_script(cfg,cwd,deffnm,is_em=False,array=None):
    s=cfg['execution']['slurm']; module=s['module']
    arr=f"#SBATCH --array={array}\n" if array else ''
    # Full GPU offload first; fallback drops update, then falls back to nonbonded GPU only.
    flags='-nb gpu' if is_em else '-nb gpu -pme gpu -bonded gpu -update gpu'
    fallback='-nb gpu' if is_em else '-nb gpu -pme gpu -bonded gpu'
    fallback2='-nb gpu'
    body=f'''#!/bin/bash\n#SBATCH --job-name=lbpacs_{deffnm}\n#SBATCH --account={s['account']}\n#SBATCH --partition={s['partition']}\n#SBATCH --gres={s['gres']}\n#SBATCH --nodes={s['nodes']}\n#SBATCH --ntasks={s['ntasks']}\n#SBATCH --cpus-per-task={s['cpus_per_task']}\n#SBATCH --time={s['time']}\n#SBATCH --output=slurm_%A_%a.out\n#SBATCH --error=slurm_%A_%a.err\n{arr}set -Eeuo pipefail\nmodule purge\nmodule load {module}\ncd "{Path(cwd).resolve()}"\nexport OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\nGMX={cfg['execution']['gmx']}\nDEFFNM="{deffnm}"\nif [[ -n "${{SLURM_ARRAY_TASK_ID:-}}" ]]; then cd "rep_$(printf '%02d' "$SLURM_ARRAY_TASK_ID")"; DEFFNM=md; fi\n$GMX mdrun -v -deffnm "$DEFFNM" -cpt 15 -ntmpi 1 -ntomp "$SLURM_CPUS_PER_TASK" {flags} || \
$GMX mdrun -v -deffnm "$DEFFNM" -cpt 15 -ntmpi 1 -ntomp "$SLURM_CPUS_PER_TASK" {fallback} || \
$GMX mdrun -v -deffnm "$DEFFNM" -cpt 15 -ntmpi 1 -ntomp "$SLURM_CPUS_PER_TASK" {fallback2}\n'''
    p=Path(cwd)/f'job_{deffnm}.slurm'; p.write_text(body); return p

def wait_slurm(jobid,cfg):
    poll=cfg['execution']['slurm']['poll_seconds']
    while True:
        r=run(['squeue','-h','-j',str(jobid)],capture=True,check=False)
        if not (r.stdout or '').strip(): break
        time.sleep(poll)
    if shutil.which('sacct'):
        r=run(['sacct','-n','-X','-j',str(jobid),'--format=State','-P'],capture=True,check=False)
        states=[x.strip().split('|')[0] for x in (r.stdout or '').splitlines() if x.strip()]
        bad=[x for x in states if not x.startswith(('COMPLETED','COMPLETING'))]
        ensure(not bad,f'SLURM job {jobid} non-success states: {bad}')

def submit_slurm(script,cfg,wait=True):
    r=run(['sbatch','--parsable',str(script)],capture=True); jid=(r.stdout or '').strip().split(';')[0]
    ensure(jid.isdigit(),f'Could not parse sbatch job id: {r.stdout}')
    log(f'Submitted SLURM job {jid}')
    if wait: wait_slurm(jid,cfg)
    return jid

def mdrun(cfg,profile,cwd,deffnm,is_em=False,wait=True):
    if profile=='local': return local_mdrun(cfg,cwd,deffnm,is_em)
    p=slurm_script(cfg,cwd,deffnm,is_em); return submit_slurm(p,cfg,wait=wait)

# ---------- equilibration ----------
def grompp(cfg,cwd,mdp,c,out,t=None,r=None,index='index.ndx',top='topol.top'):
    cwdp=Path(cwd)
    def arg(x):
        px=Path(x)
        if px.is_absolute(): return str(px)
        if px.exists(): return str(px.resolve())
        return str(x)
    cmd=[cfg['execution']['gmx'],'grompp','-f',arg(mdp),'-c',arg(c),'-p',arg(top),'-o',str(out)]
    if t: cmd += ['-t',arg(t)]
    if r: cmd += ['-r',arg(r)]
    if index and (Path(cwd)/index).exists(): cmd += ['-n',index]
    run(cmd,cwd=cwd)

def stage_equilibrate(cfg,w,profile):
    st=w/'02_system'; mark=st/'.equilibrated'
    if is_done(mark): return
    grompp(cfg,st,ROOT/'mdp'/'em.mdp','solv_ions.gro','EM.tpr',index=None); mdrun(cfg,profile,st,'EM',True)
    grompp(cfg,st,ROOT/'mdp'/'nvt.mdp','EM.gro','NVT.tpr',r='EM.gro'); mdrun(cfg,profile,st,'NVT')
    grompp(cfg,st,ROOT/'mdp'/'npt.mdp','NVT.gro','NPT.tpr',t='NVT.cpt',r='NVT.gro'); mdrun(cfg,profile,st,'NPT')
    ensure((st/'NPT.gro').exists() and (st/'NPT.xtc').exists(),'NPT outputs missing')
    done(mark)

# ---------- reference thresholds ----------
def robust_padding(values,cfg):
    v=np.asarray(values,float); med=np.median(v); mad=np.median(np.abs(v-med)); sigma=1.4826*mad
    return max(float(cfg['reference']['min_padding_A']),float(cfg['reference']['mad_multiplier'])*sigma)

def pocket_sel_from_meta(meta):
    ids=sorted(set(int(x['resid']) for x in meta['pocket_residues']))
    return 'protein and resid ' + ' '.join(map(str,ids))

def metrics_trajectory(tpr,traj,ref_tpr,ref_coord,meta,stable_fraction=0.0,pocket_indices=None):
    from MDAnalysis.lib.distances import minimize_vectors

    u=mda.Universe(str(tpr),str(traj))
    ref=mda.Universe(str(ref_tpr),str(ref_coord))

    ligsel=f"resname {meta['ligand_resname']} and not name H*"

    prot=u.select_atoms('protein')
    lig=u.select_atoms(ligsel)

    rprot=ref.select_atoms('protein')
    rlig=ref.select_atoms(ligsel)

    pock=(u.atoms[np.asarray(pocket_indices,dtype=int)]
          if pocket_indices is not None
          else u.select_atoms(pocket_sel_from_meta(meta)))

    rpock=(ref.atoms[np.asarray(pocket_indices,dtype=int)]
           if pocket_indices is not None
           else ref.select_atoms(pocket_sel_from_meta(meta)))

    ensure(len(prot)>0,'Protein selection empty in trajectory')
    ensure(len(lig)>0,'Ligand selection empty in trajectory')
    ensure(len(pock)>0,'Pocket selection empty in trajectory')
    ensure(len(rpock)==len(pock),'Reference/mobile pocket atom mismatch')
    ensure(len(lig)==len(rlig),'Ligand heavy atom mismatch for RMSD')

    mobca=u.select_atoms('protein and name CA')
    refca=ref.select_atoms('protein and name CA')
    ensure(len(mobca)==len(refca),'Protein CA mismatch')

    # Build a physically whole bound reference.
    rprot.unwrap(compound='fragments')
    rlig.unwrap(compound='fragments')

    ensure(
        ref.dimensions is not None and
        np.all(np.asarray(ref.dimensions[:3]) > 0),
        'Reference box dimensions unavailable for PBC correction'
    )

    rlig_com=rlig.center_of_mass()
    rpock_com=rpock.center_of_mass()
    rdelta=np.asarray(rlig_com-rpock_com,dtype=float).reshape(1,3)
    rdelta_mi=minimize_vectors(rdelta,ref.dimensions)[0]

    # Put reference ligand in the nearest image of its binding pocket.
    rlig.translate((rpock_com+rdelta_mi)-rlig_com)

    out=[]
    start=int(len(u.trajectory)*stable_fraction)

    for k,ts in enumerate(u.trajectory):
        if k<start:
            continue

        ensure(
            ts.dimensions is not None and
            np.all(np.asarray(ts.dimensions[:3]) > 0),
            f'Box dimensions unavailable at trajectory frame {k}'
        )

        # Reconstruct bonded protein and ligand across PBC.
        prot.unwrap(compound='fragments')
        lig.unwrap(compound='fragments')

        # Put the ligand in the periodic image nearest the physical pocket.
        lig_com=lig.center_of_mass()
        pock_com=pock.center_of_mass()

        delta=np.asarray(lig_com-pock_com,dtype=float).reshape(1,3)
        delta_mi=minimize_vectors(delta,ts.dimensions)[0]

        lig.translate((pock_com+delta_mi)-lig_com)

        # Superpose using protein C-alpha atoms only.
        # Ligand is NOT fitted to the reference ligand.
        align.alignto(
            u,
            ref,
            select='protein and name CA',
            weights='mass'
        )

        com=np.linalg.norm(
            lig.center_of_mass()-pock.center_of_mass()
        )

        rms=np.sqrt(
            np.mean(
                np.sum(
                    (lig.positions-rlig.positions)**2,
                    axis=1
                )
            )
        )

        out.append(
            (k,float(ts.time),float(com),float(rms))
        )

    return out

def stage_reference(cfg,w,meta):
    st=w/'03_reference'; mkdir(st)
    if is_done(st/'.done'): return json.loads((st/'reference.json').read_text())
    sysd=w/'02_system'
    # Save final bound reference coordinates.
    u=mda.Universe(str(sysd/'NPT.tpr'),str(sysd/'NPT.xtc')); u.trajectory[-1]
    # Freeze the pocket by atom indices from the equilibrated bound state. Protein+ligand
    # precede solvent in the topology, so deleting clashing waters later does not renumber them.
    psel=u.select_atoms(f"protein and byres (around {cfg['project']['pocket_cutoff_A']} resname {meta['ligand_resname']})")
    ensure(len(psel)>0,'No pocket atoms in final NPT bound reference')
    pocket_indices=psel.indices.astype(int).tolist()
    u.atoms.write(str(st/'bound_reference.gro'))
    vals=metrics_trajectory(sysd/'NPT.tpr',sysd/'NPT.xtc',sysd/'NPT.tpr',st/'bound_reference.gro',meta,cfg['reference']['stable_fraction'],pocket_indices)
    ensure(len(vals)>=10,'Too few stable NPT frames to derive thresholds; increase NPT output frequency/duration')
    com=[x[2] for x in vals]; rms=[x[3] for x in vals]
    com_q=float(np.percentile(com,cfg['reference']['com_percentile'])); rms_q=float(np.percentile(rms,cfg['reference']['rmsd_percentile']))
    info={'com_reference_median_A':float(np.median(com)),
          'com_threshold_A':com_q+robust_padding(com,cfg),
          'rmsd_reference_median_A':float(np.median(rms)),
          'rmsd_threshold_A':rms_q+robust_padding(rms,cfg),
          'n_frames':len(vals),
          'pocket_atom_indices':pocket_indices}
    with open(st/'bound_metrics.csv','w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['frame','time_ps','com_A','rmsd_A']); wr.writerows(vals)
    write_json(st/'reference.json',info); done(st/'.done'); log('Derived thresholds: '+json.dumps(info))
    return info

# ---------- displaced state + water clash cleanup ----------
def update_sol_count(top,water_name,count):
    lines=Path(top).read_text().splitlines(); out=[]; found=False
    for ln in lines:
        if re.match(r'^\s*'+re.escape(water_name)+r'\s+\d+\s*$',ln): out.append(f'{water_name:<16}{count}'); found=True
        else: out.append(ln)
    ensure(found,f'Could not find {water_name} count in topology')
    Path(top).write_text('\n'.join(out)+'\n')

def displace_and_clean(cfg,w,meta):
    st=w/'03_reference'; out=w/'04_pacs'; mkdir(out)
    mark=out/'.start_ready'
    if is_done(mark): return
    sysd=w/'02_system'; shutil.copy2(sysd/'topol.top',out/'topol.top'); shutil.copy2(sysd/'index.ndx',out/'index_bound.ndx')
    for p in sysd.glob('*.itp'): shutil.copy2(p,out/p.name)
    for p in sysd.glob('*.prm'): shutil.copy2(p,out/p.name)
    u=mda.Universe(str(sysd/'NPT.tpr'),str(sysd/'NPT.gro'))
    lig=u.select_atoms(f"resname {meta['ligand_resname']}"); prot=u.select_atoms('protein')
    refinfo=json.loads((w/'03_reference'/'reference.json').read_text())
    pock=u.atoms[np.asarray(refinfo['pocket_atom_indices'],dtype=int)]
    ensure(len(lig)>0 and len(pock)>0,'Displacement selections failed')
    base=lig.center_of_mass()-prot.center_of_mass(); n=np.linalg.norm(base); ensure(n>1e-6,'Ligand/protein COM vector is degenerate')
    direction=base/n; d0=np.linalg.norm(lig.center_of_mass()-pock.center_of_mass())
    original=lig.positions.copy(); chosen=None
    waters=' '.join(cfg['system']['water_resnames']); ions=' '.join(cfg['system']['ion_resnames'])
    for disp in np.arange(cfg['unbound']['initial_displacement_A'],cfg['unbound']['max_displacement_A']+0.001,cfg['unbound']['displacement_step_A']):
        lig.positions=original+direction*disp
        d=np.linalg.norm(lig.center_of_mass()-pock.center_of_mass())
        # Reject displacement into protein/ions; waters are removable.
        hard=u.select_atoms(f"(protein or resname {ions}) and around {cfg['unbound']['severe_clash_A']} resname {meta['ligand_resname']}")
        if d-d0 >= cfg['unbound']['min_ligand_pocket_com_increase_A'] and len(hard)==0:
            chosen=float(disp); break
    ensure(chosen is not None,'Could not find clash-free outward ligand displacement within configured maximum')
    # Delete only complete water residues overlapping displaced ligand.
    overlap=u.select_atoms(f"byres (resname {waters} and around {cfg['unbound']['delete_water_if_within_A']} resname {meta['ligand_resname']})")
    remove=set(overlap.resindices.tolist())
    keep=np.array([a.resindex not in remove for a in u.atoms],dtype=bool)
    u.atoms[keep].write(str(out/'displaced_raw.gro'))
    # Determine actual remaining water molecules and patch topology.
    u2=mda.Universe(str(out/'displaced_raw.gro')); water_counts={rn:len(u2.select_atoms(f'resname {rn}').residues) for rn in cfg['system']['water_resnames']}
    # GROMACS CHARMM TIP3P topology normally uses SOL; patch whichever exists in topol.top.
    toptext=read_text(out/'topol.top'); patched=False
    for rn,cnt in water_counts.items():
        if re.search(r'(?m)^\s*'+re.escape(rn)+r'\s+\d+\s*$',toptext): update_sol_count(out/'topol.top',rn,cnt); patched=True; break
    ensure(patched,'Could not identify solvent molecule name in [ molecules ]')
    write_index_from_coords(out/'displaced_raw.gro',meta,cfg,out/'index.ndx')
    grompp(cfg,out,ROOT/'mdp'/'displaced_em.mdp','displaced_raw.gro','start_em.tpr',index=None)
    # Caller will run mdrun after this function; profile is handled outside.
    write_json(out/'displacement.json',{'displacement_A':chosen,'waters_removed':len(remove),'initial_com_A':float(d0)})

def finish_displaced(cfg,w,profile,meta):
    out=w/'04_pacs'
    if is_done(out/'.start_ready'): return
    if not (out/'start_em.tpr').exists(): displace_and_clean(cfg,w,meta)
    mdrun(cfg,profile,out,'start_em',True)
    ensure((out/'start_em.gro').exists(),'Displaced-state EM failed to produce start_em.gro')
    write_index_from_coords(out/'start_em.gro',meta,cfg,out/'index.ndx')
    done(out/'.start_ready')

# ---------- PaCS ----------
def render_pacs_mdp(cfg,seed,path):
    template=read_text(ROOT/'mdp'/'pacs.mdp')
    steps=int(round(cfg['pacs']['short_md_ps']/cfg['pacs']['dt_ps']))
    stride=max(1,int(round(cfg['pacs']['frame_stride_ps']/cfg['pacs']['dt_ps'])))
    template=template.replace('@NSTEPS@',str(steps)).replace('@SEED@',str(seed)).replace('@STRIDE@',str(stride))
    Path(path).write_text(template)

def extract_frame_pbc_safe(tpr,trr,frame_index,outgro,outtrr,meta,pocket_indices):
    """
    Export one PaCS starting structure using the SAME physical PBC convention
    as the pocket-specific COM metric:

      1. reconstruct protein across PBC
      2. reconstruct ligand across PBC
      3. image ligand nearest the physical binding pocket
      4. center the Protein-Ligand complex in the periodic box
      5. wrap only solvent/ions back into the primary box
      6. preserve the simulation box vectors

    No protein/ligand fitting is performed here. Protein C-alpha fitting is
    used only inside metrics_trajectory() for RMSD validation.
    """
    from MDAnalysis.lib.distances import minimize_vectors
    from MDAnalysis.lib.mdamath import triclinic_vectors

    u=mda.Universe(str(tpr),str(trr))

    ensure(
        0 <= int(frame_index) < len(u.trajectory),
        f'Invalid trajectory frame {frame_index} for {trr}'
    )

    ts=u.trajectory[int(frame_index)]

    ensure(
        ts.dimensions is not None and
        np.all(np.asarray(ts.dimensions[:3]) > 0),
        f'Box dimensions unavailable for exported frame {frame_index}'
    )

    ligres=meta['ligand_resname']

    prot=u.select_atoms('protein')
    lig=u.select_atoms(f'resname {ligres}')

    pock=(
        u.atoms[np.asarray(pocket_indices,dtype=int)]
        if pocket_indices is not None
        else u.select_atoms(pocket_sel_from_meta(meta))
    )

    ensure(len(prot)>0,'Protein selection empty during PaCS seed export')
    ensure(len(lig)>0,'Ligand selection empty during PaCS seed export')
    ensure(len(pock)>0,'Pocket selection empty during PaCS seed export')

    # Make bonded macromolecular components physically whole.
    prot.unwrap(compound='fragments')
    lig.unwrap(compound='fragments')

    # Put ligand in the periodic image nearest the pocket.
    lig_com=lig.center_of_mass()
    pock_com=pock.center_of_mass()

    delta=np.asarray(
        lig_com-pock_com,
        dtype=float
    ).reshape(1,3)

    delta_mi=minimize_vectors(delta,ts.dimensions)[0]

    lig.translate(
        (pock_com+delta_mi)-lig_com
    )

    # Center the physically meaningful Protein-Ligand complex.
    complex_ag=u.select_atoms(
        f'protein or resname {ligres}'
    )
    ensure(
        len(complex_ag)>0,
        'Protein-Ligand complex selection empty during seed export'
    )

    box_vectors=np.asarray(
        triclinic_vectors(ts.dimensions),
        dtype=float
    )

    box_center=0.5*np.sum(box_vectors,axis=0)

    shift=box_center-complex_ag.center_of_mass()

    # Move the complete system by one common translation so solvent remains
    # spatially consistent with the selected snapshot.
    u.atoms.translate(shift)

    # Wrap only environment molecules. Do NOT independently wrap protein or
    # ligand, because that could destroy their whole/nearest-pocket geometry.
    env=u.select_atoms(
        f'not (protein or resname {ligres})'
    )

    if len(env)>0:
        env.wrap(compound='residues')

    # Preserve the original periodic box.
    u.dimensions=ts.dimensions.copy()

    outgro=Path(outgro)
    outtrr=Path(outtrr)

    with mda.Writer(str(outgro),n_atoms=len(u.atoms)) as wr:
        wr.write(u.atoms)

    # One-frame corrected TRR is retained as an auditable representation of
    # exactly the structure exported to the GRO seed.
    with mda.Writer(str(outtrr),n_atoms=len(u.atoms)) as wr:
        wr.write(u.atoms)


def load_history(csvp):
    if not Path(csvp).exists(): return []
    with open(csvp) as f: return list(csv.DictReader(f))

def append_history(csvp,row):
    exists=Path(csvp).exists()
    with open(csvp,'a',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['cycle','replica','frame','time_ps','com_A','rmsd_A','seed'])
        if not exists: wr.writeheader()
        wr.writerow(row)

def ensemble_convergence(base,current_cycle,refinfo,cfg):
    """
    Top-5 ensemble convergence gate for entry into 100-ns production.

    Convergence requires, over the complete persistence window:

      1. In every cycle, at least a majority of the Top-N snapshots
         satisfy the automatic bound COM threshold.

      2. Those same bound-like snapshots must also satisfy the automatic
         bound RMSD threshold when pose-RMSD validation is enabled.

      3. This condition must persist for convergence_window consecutive
         cycles.

      4. Ensemble stability is evaluated from the median COM of the
         Top-N snapshots in each cycle.  The range of those medians over
         the persistence window must remain within plateau_tolerance_A.

      5. After convergence, production starts from a real bound-like
         Top-N seed that satisfies BOTH COM and RMSD criteria.

    Legacy winner-only cycles cannot satisfy this convergence gate.
    """

    import math

    window = int(cfg['pacs']['convergence_window'])
    nrep = int(cfg['pacs']['replicas'])

    # For 5 replicas this gives 3/5.
    required = max(
        1,
        int(math.ceil(nrep / 2.0))
    )

    if current_cycle < window:
        return False, None, None

    cycles = list(
        range(
            current_cycle - window + 1,
            current_cycle + 1
        )
    )

    cycle_stats = []
    valid_candidates = []

    for cyc in cycles:

        cd = base / f'cycle_{cyc:03d}'
        seedfile = cd / 'selected_seeds.csv'

        # Old single-winner cycles are deliberately excluded.
        if not seedfile.exists():
            return False, None, None

        with seedfile.open() as fh:
            rows = list(csv.DictReader(fh))

        if len(rows) < nrep:
            return False, None, None

        coms = np.asarray(
            [float(r['com_A']) for r in rows],
            dtype=float
        )

        bound_count = 0

        for r in rows:

            com = float(r['com_A'])
            rms = float(r['rmsd_A'])

            good_com = (
                com <= float(refinfo['com_threshold_A'])
            )

            good_rms = (
                (not cfg['pacs']['use_pose_rmsd'])
                or
                rms <= float(refinfo['rmsd_threshold_A'])
            )

            # A snapshot is genuinely bound-like only when it satisfies
            # BOTH required criteria.
            if good_com and good_rms:

                bound_count += 1

                valid_candidates.append({
                    'cycle': int(cyc),
                    'rank': int(r['rank']),
                    'source_replica': int(r['source_replica']),
                    'source_frame': int(r['source_frame']),
                    'time_ps': float(r['time_ps']),
                    'com_A': com,
                    'rmsd_A': rms,
                    'gro': str(
                        cd / r['gro']
                    )
                })

        cycle_stats.append({
            'cycle': int(cyc),
            'bound_count': int(bound_count),
            'required_bound_count': int(required),
            'median_com_A': float(np.median(coms))
        })

        # Persistence condition:
        # every cycle in the window must contain enough bound-like seeds.
        if bound_count < required:
            return False, cycle_stats, None


    # Ensemble plateau/stability condition.
    medians = np.asarray(
        [s['median_com_A'] for s in cycle_stats],
        dtype=float
    )

    median_range = float(
        medians.max() - medians.min()
    )

    if median_range > float(
        cfg['pacs']['plateau_tolerance_A']
    ):
        return False, cycle_stats, None


    if not valid_candidates:
        return False, cycle_stats, None


    # Choose a real production structure.
    #
    # COM remains the primary physical ranking coordinate.
    # RMSD is a required validation criterion, not a weighted CV.
    best = min(
        valid_candidates,
        key=lambda x: (
            x['com_A'],
            x['rmsd_A'],
            x['cycle'],
            x['rank']
        )
    )

    ensure(
        Path(best['gro']).exists(),
        f"Selected production seed missing: {best['gro']}"
    )

    return True, cycle_stats, best


def plot_history(hist,refinfo,out):
    try:
        import matplotlib.pyplot as plt
        c=[int(r['cycle']) for r in hist]; d=[float(r['com_A']) for r in hist]
        plt.figure(); plt.plot(c,d,marker='o'); plt.axhline(refinfo['com_threshold_A'],linestyle='--'); plt.xlabel('PaCS cycle'); plt.ylabel('Best ligand-pocket COM distance (A)'); plt.tight_layout(); plt.savefig(out,dpi=160); plt.close()
    except Exception as e: log(f'WARNING: convergence plot failed: {e}')



def run_pacs(cfg,w,profile,meta,refinfo):
    base=w/'04_pacs'
    histp=base/'convergence.csv'
    hist=load_history(histp)

    if (base/'CONVERGED').exists():
        return True,base/'final_converged.gro'

    nrep=int(cfg['pacs']['replicas'])

    # ------------------------------------------------------------------
    # Resume/start policy
    # ------------------------------------------------------------------
    #
    # Fresh PaCS:
    #     every replica starts from displaced-EM state.
    #
    # New diversified cycles:
    #     replica i starts from seed_i.gro generated by previous cycle.
    #
    # Legacy single-winner cycles:
    #     the first new cycle uses winner.gro once as a transition.
    #     That cycle generates five diversified ranked seeds, so all later
    #     cycles use independent starting snapshots.
    #
    if not hist:
        starts=[
            base/'start_em.gro'
            for _ in range(nrep)
        ]
        start_cycle=1

    else:
        prev_cycle=int(hist[-1]['cycle'])
        prev=base/f'cycle_{prev_cycle:03d}'

        seed_starts=[
            prev/f'seed_{rep:02d}.gro'
            for rep in range(1,nrep+1)
        ]

        if all(s.exists() for s in seed_starts):
            starts=seed_starts

            log(
                f'PaCS resume: using {nrep} diversified ranked seeds '
                f'from cycle {prev_cycle}'
            )

        else:
            legacy=prev/'winner.gro'

            ensure(
                legacy.exists(),
                f'Missing previous PaCS state: {legacy}'
            )

            starts=[
                legacy
                for _ in range(nrep)
            ]

            log(
                f'PaCS transition from legacy single-winner cycle '
                f'{prev_cycle}: cycle {prev_cycle+1} uses winner.gro '
                f'once; subsequent cycles use Top-{nrep} ranked seeds.'
            )

        start_cycle=prev_cycle+1


    # ------------------------------------------------------------------
    # PaCS cycles
    # ------------------------------------------------------------------
    for cyc in range(
        start_cycle,
        int(cfg['pacs']['max_cycles'])+1
    ):

        cd=base/f'cycle_{cyc:03d}'
        mkdir(cd)

        # --------------------------------------------------------------
        # Prepare replicas
        # --------------------------------------------------------------
        incomplete_reps=[]

        for rep in range(1,nrep+1):

            rd=cd/f'rep_{rep:02d}'
            mkdir(rd)

            md_complete=(
                (rd/'md.xtc').exists()
                and (rd/'md.gro').exists()
                and (rd/'md.log').exists()
                and 'Finished mdrun' in (rd/'md.log').read_text(
                    errors='ignore'
                )
            )

            if md_complete:
                log(
                    f'PaCS cycle {cyc} replica {rep}: '
                    f'existing completed MD detected; skipping rerun'
                )
                continue

            incomplete_reps.append(rep)

            md_seed=(
                int(cfg['pacs']['random_seed_base'])
                +cyc*1000
                +rep
            )

            render_pacs_mdp(
                cfg,
                md_seed,
                rd/'pacs.mdp'
            )

            for name in ['topol.top','index.ndx']:
                target=rd/name

                if not target.exists():
                    target.symlink_to(
                        (base/name).resolve()
                    )

            for pth in base.glob('*.itp'):
                target=rd/pth.name

                if not target.exists():
                    target.symlink_to(
                        pth.resolve()
                    )

            for pth in base.glob('*.prm'):
                target=rd/pth.name

                if not target.exists():
                    target.symlink_to(
                        pth.resolve()
                    )

            rep_start=Path(
                starts[rep-1]
            ).resolve()

            ensure(
                rep_start.exists(),
                f'Missing starting structure for '
                f'cycle {cyc} replica {rep}: {rep_start}'
            )

            log(
                f'PaCS cycle {cyc} replica {rep}: '
                f'start={rep_start}'
            )

            grompp(
                cfg,
                rd,
                rd/'pacs.mdp',
                rep_start,
                'md.tpr',
                index='index.ndx',
                top='topol.top'
            )


        # --------------------------------------------------------------
        # Execute only replicas that are not already complete
        # --------------------------------------------------------------
        if incomplete_reps:

            if profile=='slurm_gpu':

                array_spec=','.join(
                    str(rep) for rep in incomplete_reps
                )

                jobscript=slurm_script(
                    cfg,
                    cd,
                    'pacs_array',
                    False,
                    array=array_spec
                )

                submit_slurm(
                    jobscript,
                    cfg,
                    wait=True
                )

            else:

                for rep in incomplete_reps:
                    local_mdrun(
                        cfg,
                        cd/f'rep_{rep:02d}',
                        'md'
                    )

        else:
            log(
                f'PaCS cycle {cyc}: all {nrep} replicas already '
                f'complete; proceeding directly to trajectory analysis'
            )


        # --------------------------------------------------------------
        # GLOBAL all-frame analysis
        # --------------------------------------------------------------
        #
        # Important:
        #   - every post-t=0 frame from every replica participates
        #   - ranking criterion = pocket-specific PBC-aware COM ONLY
        #   - RMSD is recorded but NOT used as a ranking weight
        #
        all_candidates=[]

        for rep in range(1,nrep+1):

            rd=cd/f'rep_{rep:02d}'

            vals=metrics_trajectory(
                rd/'md.tpr',
                rd/'md.xtc',
                w/'02_system'/'NPT.tpr',
                w/'03_reference'/'bound_reference.gro',
                meta,
                0.0,
                refinfo['pocket_atom_indices']
            )

            ensure(
                vals,
                f'No trajectory frames in '
                f'cycle {cyc} replica {rep}'
            )

            eligible=[
                x for x in vals
                if float(x[1]) > 1.0e-6
            ]

            ensure(
                eligible,
                f'No post-t=0 frames in '
                f'cycle {cyc} replica {rep}'
            )

            for frame,tps,com,rms in eligible:

                all_candidates.append({
                    'replica':int(rep),
                    'frame':int(frame),
                    'time_ps':float(tps),
                    'com_A':float(com),
                    'rmsd_A':float(rms),
                })


        ensure(
            len(all_candidates)>=nrep,
            f'Only {len(all_candidates)} eligible PaCS frames '
            f'available for Top-{nrep} selection in cycle {cyc}'
        )


        # --------------------------------------------------------------
        # Global COM ranking
        # --------------------------------------------------------------
        #
        # Each (replica, frame) pair is intrinsically a distinct snapshot.
        # No artificial RMSD/diversity threshold is introduced, so the
        # physical CV definition remains unchanged.
        #
        all_candidates.sort(
            key=lambda x:x['com_A']
        )

        selected=all_candidates[:nrep]

        selected_keys={
            (c['replica'],c['frame'])
            for c in selected
        }


        # --------------------------------------------------------------
        # Full ranking log for EVERY frame
        # --------------------------------------------------------------
        rankfile=cd/'all_frames_ranked.csv'

        with rankfile.open(
            'w',
            newline=''
        ) as fh:

            fields=[
                'rank',
                'replica',
                'frame',
                'time_ps',
                'com_A',
                'rmsd_A',
                'selected'
            ]

            wr=csv.DictWriter(
                fh,
                fieldnames=fields
            )

            wr.writeheader()

            for rank,c in enumerate(
                all_candidates,
                1
            ):

                wr.writerow({
                    'rank':rank,
                    'replica':c['replica'],
                    'frame':c['frame'],
                    'time_ps':f"{c['time_ps']:.6f}",
                    'com_A':f"{c['com_A']:.6f}",
                    'rmsd_A':f"{c['rmsd_A']:.6f}",
                    'selected':(
                        1
                        if (
                            c['replica'],
                            c['frame']
                        ) in selected_keys
                        else 0
                    )
                })


        # --------------------------------------------------------------
        # Export Top-N starting snapshots with explicit PBC handling
        # --------------------------------------------------------------
        next_starts=[]
        seed_records=[]

        for rank,c in enumerate(
            selected,
            1
        ):

            source_rep=int(
                c['replica']
            )

            src=cd/f'rep_{source_rep:02d}'

            seedgro=cd/f'seed_{rank:02d}.gro'
            seedtrr=cd/f'seed_{rank:02d}_full.trr'

            extract_frame_pbc_safe(
                src/'md.tpr',
                src/'md.trr',
                int(c['frame']),
                seedgro,
                seedtrr,
                meta,
                refinfo['pocket_atom_indices']
            )

            ensure(
                seedgro.exists(),
                f'Failed to export {seedgro}'
            )

            next_starts.append(
                seedgro
            )

            seed_records.append({
                'rank':rank,
                'next_replica':rank,
                'source_replica':source_rep,
                'source_frame':int(c['frame']),
                'time_ps':float(c['time_ps']),
                'com_A':float(c['com_A']),
                'rmsd_A':float(c['rmsd_A']),
                'gro':seedgro.name,
                'trr':seedtrr.name
            })


        # Compact selected-seed table.
        with (
            cd/'selected_seeds.csv'
        ).open(
            'w',
            newline=''
        ) as fh:

            fields=[
                'rank',
                'next_replica',
                'source_replica',
                'source_frame',
                'time_ps',
                'com_A',
                'rmsd_A',
                'gro'
            ]

            wr=csv.DictWriter(
                fh,
                fieldnames=fields
            )

            wr.writeheader()

            for s in seed_records:
                wr.writerow({
                    k:s[k]
                    for k in fields
                })


        write_json(
            cd/'seeds.json',
            {
                'cycle':cyc,
                'ranking_metric':
                    'pocket-specific PBC-aware COM distance',
                'ranking_order':
                    'ascending',
                'rmsd_role':
                    'recorded and used for convergence validation only',
                'frame_zero':
                    'excluded',
                'selection':
                    f'global Top-{nrep} distinct replica/frame snapshots',
                'export':
                    'protein whole; ligand whole; ligand nearest pocket; '
                    'Protein-Ligand complex centered; solvent/ions wrapped',
                'seeds':seed_records
            }
        )


        # --------------------------------------------------------------
        # Rank 1 remains official winner for backward compatibility,
        # history, plotting and strict convergence.
        # --------------------------------------------------------------
        best=selected[0]

        rep=int(best['replica'])
        frame=int(best['frame'])
        tps=float(best['time_ps'])
        com=float(best['com_A'])
        rms=float(best['rmsd_A'])

        md_seed=(
            int(cfg['pacs']['random_seed_base'])
            +cyc*1000
            +rep
        )

        shutil.copy2(
            cd/'seed_01.gro',
            cd/'winner.gro'
        )

        shutil.copy2(
            cd/'seed_01_full.trr',
            cd/'winner_full.trr'
        )

        row={
            'cycle':cyc,
            'replica':rep,
            'frame':frame,
            'time_ps':f'{tps:.6f}',
            'com_A':f'{com:.6f}',
            'rmsd_A':f'{rms:.6f}',
            'seed':md_seed
        }

        append_history(
            histp,
            row
        )

        hist.append({
            k:str(v)
            for k,v in row.items()
        })

        plot_history(
            hist,
            refinfo,
            base/'convergence.png'
        )

        write_json(
            cd/'winner.json',
            row
        )

        log(
            f'PaCS cycle {cyc}: '
            f'global winner rep={rep} '
            f'frame={frame} '
            f't={tps:.3f} ps '
            f'COM={com:.3f} A '
            f'RMSD={rms:.3f} A'
        )

        for s in seed_records:

            log(
                f"  Top-{s['rank']} seed -> next replica "
                f"{s['next_replica']}: "
                f"source rep={s['source_replica']} "
                f"frame={s['source_frame']} "
                f"t={s['time_ps']:.3f} ps "
                f"COM={s['com_A']:.3f} A "
                f"RMSD={s['rmsd_A']:.3f} A"
            )


        # --------------------------------------------------------------
        # Cleanup remains optional and occurs ONLY after:
        #   - all frames were ranked
        #   - all Top-N seeds were safely exported
        #   - ranking metadata was written
        # --------------------------------------------------------------
        if cfg['pacs']['cleanup_losers']:

            for rr in range(1,nrep+1):

                if rr==rep:
                    continue

                for ext in (
                    'xtc',
                    'trr',
                    'edr',
                    'cpt',
                    'tpr'
                ):

                    q=(
                        cd/f'rep_{rr:02d}'
                        /f'md.{ext}'
                    )

                    if q.exists():
                        q.unlink()


        # --------------------------------------------------------------
        # ORIGINAL strict convergence logic remains untouched.
        # --------------------------------------------------------------
        ensemble_ok,ensemble_stats,production_seed = ensemble_convergence(
            base,
            cyc,
            refinfo,
            cfg
        )

        if ensemble_stats is not None:
            write_json(
                cd/'ensemble_convergence.json',
                {
                    'cycle':cyc,
                    'window':int(cfg['pacs']['convergence_window']),
                    'replicas':nrep,
                    'required_bound_per_cycle':int(
                        np.ceil(nrep/2.0)
                    ),
                    'com_threshold_A':float(
                        refinfo['com_threshold_A']
                    ),
                    'rmsd_threshold_A':float(
                        refinfo['rmsd_threshold_A']
                    ),
                    'plateau_tolerance_A':float(
                        cfg['pacs']['plateau_tolerance_A']
                    ),
                    'cycle_statistics':ensemble_stats,
                    'converged':bool(ensemble_ok),
                    'production_seed':production_seed
                }
            )

        if ensemble_ok:

            prodsrc=Path(
                production_seed['gro']
            )

            shutil.copy2(
                prodsrc,
                base/'final_converged.gro'
            )

            write_json(
                base/'production_seed.json',
                production_seed
            )

            (
                base/'CONVERGED'
            ).write_text(
                f"cycle={cyc}\n"
                f"method=top5_ensemble\n"
                f"source_cycle={production_seed['cycle']}\n"
                f"source_rank={production_seed['rank']}\n"
                f"com_A={production_seed['com_A']:.6f}\n"
                f"rmsd_A={production_seed['rmsd_A']:.6f}\n"
            )

            log(
                f"ENSEMBLE CONVERGED at cycle {cyc}: "
                f"production seed cycle="
                f"{production_seed['cycle']} "
                f"rank={production_seed['rank']} "
                f"COM={production_seed['com_A']:.3f} A "
                f"RMSD={production_seed['rmsd_A']:.3f} A"
            )

            return (
                True,
                base/'final_converged.gro'
            )


        # Critical propagation change:
        # next cycle receives Top-N independent ranked structures.
        starts=next_starts


    # ------------------------------------------------------------------
    # Max cycles reached without strict convergence.
    # Production remains blocked exactly as before.
    # ------------------------------------------------------------------
    if hist:

        b=min(
            hist,
            key=lambda r:float(r['com_A'])
        )

        src=(
            base
            /f"cycle_{int(b['cycle']):03d}"
            /'winner.gro'
        )

        shutil.copy2(
            src,
            base/'best_state.gro'
        )

    (
        base/'MAX_CYCLES_REACHED'
    ).write_text(
        'CONVERGED=false\n'
    )

    return False,None

# ---------- production launch only ----------
def render_prod_mdp(cfg,path):
    t=read_text(ROOT/'mdp'/'production_100ns.mdp')
    steps=int(round(cfg['production']['length_ns']*1000/cfg['production']['dt_ps']))
    stride=max(1,int(round(cfg['production']['trajectory_stride_ps']/cfg['production']['dt_ps'])))
    Path(path).write_text(t.replace('@NSTEPS@',str(steps)).replace('@XTCSTRIDE@',str(stride)))

def launch_production(cfg,w,profile,state):
    st=w/'05_production'; mkdir(st)
    if (st/'SUBMITTED').exists() or (st/'LOCAL_STARTED').exists(): return
    base=w/'04_pacs'
    for n in ['topol.top','index.ndx']:
        shutil.copy2(base/n,st/n)
    for p in base.glob('*.itp'): shutil.copy2(p,st/p.name)
    for p in base.glob('*.prm'): shutil.copy2(p,st/p.name)
    shutil.copy2(state,st/'start.gro'); render_prod_mdp(cfg,st/'production.mdp')
    grompp(cfg,st,st/'production.mdp','start.gro','production.tpr',index='index.ndx',top='topol.top')
    if profile=='slurm_gpu':
        p=slurm_script(cfg,st,'production',False)
        jid=submit_slurm(p,cfg,wait=False); (st/'SUBMITTED').write_text(f'{jid}\n'); log(f'100-ns production submitted as SLURM job {jid}; pipeline intentionally does not analyze it.')
    else:
        # CPU laptop: do not accidentally occupy the laptop for 100 ns. TPR is fully prepared.
        cmd=f"{cfg['execution']['gmx']} mdrun -deffnm production -ntmpi 1 -ntomp {cfg['execution']['local_threads']}"
        (st/'LOCAL_STARTED').write_text('NOT_AUTO_RUN_ON_CPU\n'+cmd+'\n')
        log('Local CPU profile: production.tpr prepared but 100 ns was not auto-run to avoid an accidental multi-day CPU job.')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--profile',choices=['local','slurm_gpu'],required=True); args=ap.parse_args()
    cfg=load_cfg(args.config); w=ROOT/cfg['project']['workdir']; mkdir(w)
    ff=preflight(cfg,args.profile)
    meta=stage_detect(cfg,w)
    lig=stage_param(cfg,w,meta)
    stage_build(cfg,w,meta,lig,ff)
    stage_equilibrate(cfg,w,args.profile)
    ref=stage_reference(cfg,w,meta)
    finish_displaced(cfg,w,args.profile,meta)
    ok,state=run_pacs(cfg,w,args.profile,meta,ref)
    if ok: launch_production(cfg,w,args.profile,state)
    else: log('PaCS reached max cycles without strict convergence; 100-ns production was NOT launched.')

if __name__=='__main__':
    try: main()
    except PipelineError as e:
        print('\nFATAL:',e,file=sys.stderr); sys.exit(2)
