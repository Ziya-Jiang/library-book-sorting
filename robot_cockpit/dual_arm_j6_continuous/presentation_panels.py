"""Read-only English presentation images, using the saved left-wrist exposure."""
import json
from pathlib import Path
import cv2
import numpy as np
from stack_overlay import draw_books


def banner(rgb, lines):
    height=12+26*len(lines)
    header=np.full((height,rgb.shape[1],3),(20,25,32),np.uint8)
    image=np.vstack([header,rgb])
    for row,line in enumerate(lines):
        if not line.isascii():raise ValueError('Presentation text must be English/ASCII')
        cv2.putText(image,line,(12,26+26*row),cv2.FONT_HERSHEY_SIMPLEX,.58,
                    (235,240,250),1,cv2.LINE_AA)
    return image


def placeholder(title, detail='Waiting for this run', shape=(480,848)):
    return banner(np.full((*shape,3),(22,28,36),np.uint8),[title,detail])


def segmentation(rgb, reference):
    image=rgb.copy();mask=np.zeros(rgb.shape[:2],np.uint8)
    polygon=np.rint(reference['polygon_px']).astype(np.int32)
    cv2.fillPoly(mask,[polygon],1)
    on=mask.astype(bool);image[on]=(.6*image[on]+.4*np.array([255,210,40])).astype(np.uint8)
    cv2.polylines(image,[polygon],True,(255,210,40),2,cv2.LINE_AA)
    edge=np.rint(reference['far_edge']['edge_12_px']).astype(np.int32)
    cv2.line(image,tuple(edge[0]),tuple(edge[1]),(255,80,210),3,cv2.LINE_AA)
    for i,point in enumerate(edge,1):
        cv2.circle(image,tuple(point),5,(255,80,210),-1,cv2.LINE_AA)
        label=(min(int(point[0])+8,image.shape[1]-24),max(24,int(point[1])-8))
        cv2.putText(image,str(i),label,cv2.FONT_HERSHEY_SIMPLEX,.8,
                    (255,255,255),2,cv2.LINE_AA)
    return banner(image,['LEFT WRIST | Spine + edge 1-2',
                         'Yellow: robot-facing spine   Pink: far edge 1-2'])


def grasp_marker(rgb, depth, xyz, reference):
    """Mark an actual depth sample nearest the measured spine center, not image center."""
    polygon=np.rint(reference['polygon_px']).astype(np.int32)
    mask=np.zeros(depth.shape,np.uint8);cv2.fillPoly(mask,[polygon],1)
    valid=cv2.erode(mask,np.ones((3,3),np.uint8)).astype(bool)
    valid &= np.isfinite(depth)&(depth>0)&(depth<=5000)
    uv=np.column_stack(np.nonzero(valid))[:,::-1]
    xyz=np.asarray(xyz,float);center=np.asarray(reference['center_base_mm'],float)
    if (xyz.shape!=(len(uv),3) or len(uv)<30 or center.shape!=(3,)
            or not np.isfinite(xyz).all() or not np.isfinite(center).all()):
        raise ValueError('Saved spine depth points do not match the exposure mask')
    index=int(np.argmin(np.linalg.norm(xyz-center,axis=1)))
    point=tuple(uv[index].astype(int));target_y=float(center[1])
    image=rgb.copy()
    cv2.polylines(image,[polygon],True,(255,210,40),1,cv2.LINE_AA)
    cv2.circle(image,point,14,(0,0,0),5,cv2.LINE_AA)
    cv2.circle(image,point,14,(80,255,170),2,cv2.LINE_AA)
    cv2.drawMarker(image,point,(80,255,170),cv2.MARKER_CROSS,40,2,cv2.LINE_AA)
    cv2.putText(image,f'Y {target_y:+.2f} mm',
                (max(4,min(point[0]+24,image.shape[1]-190)),max(88,min(point[1]-20,image.shape[0]-10))),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(image,f'Y {target_y:+.2f} mm',
                (max(4,min(point[0]+24,image.shape[1]-190)),max(88,min(point[1]-20,image.shape[0]-10))),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(80,255,170),1,cv2.LINE_AA)
    image=banner(image,[f'GRASP TARGET | Base Y = {target_y:+.2f} mm',
                        'Marker: measured depth point nearest spine center'])
    return image,dict(target_grasp_y_mm=target_y,marker_uv_px=list(map(int,point)),
                      marker_measured_base_mm=xyz[index].tolist(),
                      marker_policy='nearest_measured_point_to_spine_center',
                      frame_token=reference['frame_token'])


def render_left_panels(out, reference):
    out=Path(out)
    with np.load(out/'left_wrist_frame.npz',allow_pickle=False) as data:
        rgb=cv2.cvtColor(data['bgr'],cv2.COLOR_BGR2RGB);depth=data['depth']
        header=json.loads(str(data['header']))
    if header['token']!=reference['frame_token']:
        raise ValueError('Preview reference belongs to another exposure')
    xyz=np.load(out/'left_wrist_spine_base_points_mm.npy',allow_pickle=False)
    marked,info=grasp_marker(rgb,depth,xyz,reference)
    return dict(segmentation=segmentation(rgb,reference),
                book_count=draw_books(rgb,reference['stack_books'],reference['stack_book_count'],reference['stack_lift_mm']),
                grasp_y=marked),info
