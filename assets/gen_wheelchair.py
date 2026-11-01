# 本ファイルは構成を説明するための骨組みである。報酬の重み・接触条件・終了条件の値・参照生成のパラメータは非公開(None のプレースホルダ)で、このままでは学習結果を再現できない。© 2026 okakei. All rights reserved.
# This file is a skeleton that describes the structure only. Reward weights, contact conditions, termination values and reference-generation parameters are withheld (None placeholders), so the learned results cannot be reproduced as is. © 2026 okakei. All rights reserved.
# 市販の跳ね上げ式手動車いすの 3Dモデル生成 (trimesh)。公開カタログ寸法から作る。
# フットレスト=足板が「横に・内向きに」跳ね上がる(前後軸まわりに横倒しで上へ)。前折りでも横スイングでもない。
import os, numpy as np, trimesh, shutil
from trimesh.creation import box, cylinder, annulus, torus
from trimesh.transformations import rotation_matrix as ROT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
SEAT_H=0.500; SEAT_W=0.400; SEAT_D=0.370; ARM_H=0.220; BACK_H=0.400
REAR_D=0.508; CAST_D=0.152; FRM=SEAT_H-0.03  # 後輪20in=50.8cm / 前輪6in=15.24cm
def Rx(d): return ROT(np.radians(d),[1,0,0])
def Ry(d): return ROT(np.radians(d),[0,1,0])
def Tr(x=0,y=0,z=0):
    m=np.eye(4); m[:3,3]=[x,y,z]; return m
FRAME=[190,192,196,255]; FAB=[28,28,32,255]; TIRE=[18,18,18,255]
RIM=[205,208,212,255]; PAD=[38,38,44,255]; PLATE=[60,62,70,255]
PARTS=[]
def add(mesh,color,M=None,name=None):
    if M is not None: mesh.apply_transform(M)
    mesh.visual.face_colors=color; PARTS.append((name or f'p{len(PARTS)}',mesh))
def tube(p1,p2,r,color=FRAME):
    add(cylinder(radius=r,segment=(np.array(p1),np.array(p2))),color)
def wheel(cx,cy,cz,Rt,minor,spokes=12,handrim=True):
    add(torus(Rt-minor,minor),TIRE,Tr(cx,cy,cz)@Rx(90),'tire')
    add(annulus(r_min=Rt*0.16,r_max=Rt-minor-0.005,height=0.008),RIM,Tr(cx,cy,cz)@Rx(90),'rim')
    add(cylinder(radius=Rt*0.10,height=0.05),RIM,Tr(cx,cy,cz)@Rx(90),'hub')
    rin=Rt*0.11; rout=Rt-minor-0.01; mid=(rin+rout)/2; ln=rout-rin
    for k in range(spokes):
        th=360.0*k/spokes; px=cx+mid*np.cos(np.radians(th)); pz=cz+mid*np.sin(np.radians(th))
        add(cylinder(radius=0.004,height=ln),RIM,Tr(px,cy,pz)@Ry(90-th))
    if handrim: add(torus(Rt-minor-0.03,0.010),RIM,Tr(cx,cy+np.sign(cy)*0.03,cz)@Rx(90),'handrim')

def legrest(s, flip_deg):
    # 前脚を前方かつ下へ大きく伸ばす(公式 全長95cm)。足板は長辺が内向きの長方形で、前後軸まわりに横倒しで上へ。
    py=s*(SEAT_W/2-0.02)
    tube([0.20,py,FRM],[0.44,py,0.22],0.013)              # 前脚(上段:前方下へ)
    tube([0.42,py,0.22],[0.525,py,0.09],0.013)            # 前脚(下段:足板へ)
    add(box([0.03,0.05,0.03]),FRAME,Tr(0.53,py,0.085))    # 足板ヒンジ(脚先端・前後軸)
    FWD,INW=0.09,0.18   # 足板=長方形。前後(短)0.09 × 内向き(長)0.18 ← 一番長い辺が内向き
    # 展開=脚先(外側py)から内側へ水平に長い / flip=前後軸まわりに横倒しで上へ(内向き)
    plate=Tr(0.55,py,0.075)@Rx(-s*flip_deg)@Tr(0,-s*INW/2,0)
    add(box([FWD,INW,0.012]),PLATE,plate)

def build(flip_deg):
    PARTS.clear()
    add(box([SEAT_D,SEAT_W,0.03]),FAB,Tr(0,0,SEAT_H-0.015),'seat')
    add(box([0.03,SEAT_W,BACK_H]),FAB,Tr(-SEAT_D/2,0,SEAT_H)@Ry(-8)@Tr(0,0,BACK_H/2),'backrest')
    for s in (1,-1):
        y=s*(SEAT_W/2+0.005)
        add(box([0.28,0.05,0.03]),PAD,Tr(0.01,y,SEAT_H+ARM_H-0.015))
        tube([0.13,y,SEAT_H+ARM_H-0.03],[0.13,y,SEAT_H-0.02],0.010)
        tube([-0.10,y,SEAT_H+ARM_H-0.03],[-0.10,y,SEAT_H-0.02],0.010)
    rear_y=0.265   # 公式 全幅61cm に合わせ後輪を外へ
    for s in (1,-1): wheel(-0.06,s*rear_y,REAR_D/2,REAR_D/2,0.024,12,True)
    for s in (1,-1):
        y=s*SEAT_W/2
        wheel(0.26,y,CAST_D/2,CAST_D/2,0.018,0,False)
        add(box([0.03,0.012,0.14]),FRAME,Tr(0.26,y-0.028,CAST_D/2+0.06))
        add(box([0.03,0.012,0.14]),FRAME,Tr(0.26,y+0.028,CAST_D/2+0.06))
        tube([0.26,y,CAST_D/2+0.12],[0.26,y,FRM],0.012)
    for s in (1,-1):
        y=s*(SEAT_W/2+0.005)
        tube([SEAT_D/2,y,FRM],[-SEAT_D/2,y,FRM],0.012)
        tube([0.26,y,0.12],[-0.06,s*rear_y,REAR_D/2],0.014)
        BT=SEAT_H+BACK_H-0.015   # 骨組み上端(実測全高が座高50+背40=90になるよう)
        tube([-SEAT_D/2+0.01,y,FRM],[-SEAT_D/2+0.01,y*0.9,BT],0.012)
        top=[-SEAT_D/2+0.01,y*0.9,BT]; grip=[-SEAT_D/2-0.09,y*0.9,BT-0.02]  # 押手
        tube(top,grip,0.013); add(box([0.10,0.026,0.026]),PAD,Tr(grip[0]-0.03,grip[1],grip[2]))
    tube([0.10,SEAT_W/2,FRM-0.17],[-0.10,-SEAT_W/2,FRM-0.15],0.011)
    tube([0.10,-SEAT_W/2,FRM-0.17],[-0.10,SEAT_W/2,FRM-0.15],0.011)
    for s in (1,-1): legrest(s,flip_deg)
    return trimesh.util.concatenate([m for _,m in PARTS])

for suffix,fl in [('down',0),('up',95)]:
    comb=build(fl)
    comb.export(os.path.join(OUT,f'wheelchair_{suffix}.obj'))
    comb.export(os.path.join(OUT,f'wheelchair_{suffix}.stl'))
    glb=comb.copy(); glb.apply_transform(Rx(-90)); glb.export(os.path.join(OUT,f'wheelchair_{suffix}.glb'))
    bb=comb.bounds; sz=bb[1]-bb[0]
    print(f'[{suffix} flip={fl}] LxWxH=%.3f x %.3f x %.3f'%(sz[0],sz[1],sz[2]))
for ext in ('glb','obj','stl'): shutil.copy(os.path.join(OUT,f'wheelchair_down.{ext}'),os.path.join(OUT,f'wheelchair.{ext}'))
print('done @',OUT)
