"""Head-only book-width fallback. Never supplies wrist features or a spine axis."""
import numpy as np
import json
from pathlib import Path


def load_approach_limits():
    return json.loads(Path(__file__).with_name('baseline_head_approach.json').read_text())


def book_midpoint(mask, depth, intrinsic, score):
    mask=np.asarray(mask,bool);depth=np.asarray(depth,float);k=np.asarray(intrinsic,float)
    if mask.ndim!=2 or mask.shape!=depth.shape or k.shape!=(3,3):
        raise ValueError('Invalid whole-book RGB-D shape')
    ys,xs=np.nonzero(mask)
    if not len(xs):raise ValueError('Empty whole-book mask')
    # Width midpoint of the visible book at its central occupied image row,
    # not the distribution of text/features or a fabricated spine midline.
    row=int(ys[np.argmin(abs(ys-(ys.min()+ys.max())/2))])
    cols=np.flatnonzero(mask[row]);u=float((cols[0]+cols[-1])/2);v=float(row)
    xx,yy=np.meshgrid(np.arange(mask.shape[1]),np.arange(mask.shape[0]))
    local=mask&(abs(xx-u)<=4)&(abs(yy-v)<=4)&np.isfinite(depth)&(depth>0)
    samples=depth[local]
    if len(samples)<3:raise ValueError('No measured depth at whole-book width midpoint')
    if np.median(samples)>20:samples=samples*.001
    ray=np.linalg.solve(k,np.array([u,v,1.]))
    if not np.isfinite(ray).all() or ray[2]<=0:raise ValueError('Invalid midpoint backprojection')
    z=float(np.median(samples))
    return dict(mask=mask,box=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],np.float32),
        uv=[u,v],point_camera_m=ray/ray[2]*z,depth_m=z,depth_samples=len(samples),
        depth_p10_p90_mm=(np.percentile(samples,[10,90])*1000).tolist(),score=float(score),
        center_kind='whole_book_width_midpoint',selection_source='sam31_book_width_fallback',
        spine_3d_status='unavailable: head book-width midpoint only',
        j7_skip_reason='整书宽度中点仅用于Y粗对齐；书脊中线与特征由腕部重新提取',
        _head_full_book_mask=mask)


def add_missing_books(candidates, books, depth, intrinsic, report=None):
    result=list(candidates);notes=[]
    for score,mask,_ in books:
        mask=np.asarray(mask,bool)
        if not np.isfinite(score) or score<.1:continue
        matched=[c for c in candidates if np.count_nonzero(np.asarray(c['mask'],bool)&mask)
                 /max(1,np.count_nonzero(c['mask']))>=.7]
        if matched:continue
        try:result.append(book_midpoint(mask,depth,intrinsic,score))
        except (ValueError,np.linalg.LinAlgError) as error:notes.append(str(error))
    if report is not None:
        report.update(book_width_fallbacks=len(result)-len(candidates),book_width_rejected=notes)
    return result


def prealignment_target(start, parameters, limits):
    """At most one short +X move, ending before the estimated book surface."""
    start=np.asarray(start,float)
    maximum=float(limits['max_forward_mm']);clearance=float(limits['surface_clearance_mm'])
    if not np.isfinite([maximum,clearance]).all() or not 0<=maximum<=50 or clearance<50:
        raise ValueError('Invalid baseline prealignment limits')
    stop=float(parameters['book_surface_x_mm'])-float(parameters['fingertip_x_offset_mm'])-clearance
    end=min(float(start[0])+maximum,stop,float(parameters['grasp_tcp_x_mm']))
    if end<=start[0]:return None
    target=start.copy();target[0]=end
    return target
