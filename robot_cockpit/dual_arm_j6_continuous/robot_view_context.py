"""Same-frame robot-base direction evidence for the continuous J6 RGB request.

Uses the calibrated SDK ray table (including lens distortion), not screen-side
heuristics. This is a visual hint, never a replacement for measured RGB-D.
"""
import cv2
import numpy as np


def direction_context(bgr, depth, rays, transform):
    h, w = depth.shape
    normalized = rays[..., :2] / rays[..., 2:3]
    overlay = bgr.copy()
    anchors = []
    # Keep the top-right book area clear; provide three local perspective hints.
    positions=[(w//3,h//3),(w//2,h//2),(2*w//3,2*h//3)]
    valid_v,valid_u=np.where(np.isfinite(depth) & (depth>0) & (depth<=5000))
    if len(valid_u):
        i=len(valid_u)//2
        positions.append((int(valid_u[i]),int(valid_v[i])))
    for u, v in positions:
        patch = depth[max(0,v-8):v+9, max(0,u-8):u+9]
        values = patch[np.isfinite(patch) & (patch > 0) & (patch <= 5000)]
        if len(values) < 10:
            continue
        origin = rays[v,u].astype(float) * float(np.median(values)) * .001
        directions = {}
        for name, axis, color in [('-X ROBOT', [-1.,0.,0.], (0,255,255)),
                                  ('+Y', [0.,1.,0.], (255,255,0))]:
            # Camera-to-base R: a base displacement expressed in camera is R.T d.
            endpoint = origin + transform[:3,:3].T @ (np.array(axis)*.025)
            if endpoint[2] <= 0:
                continue
            xy = endpoint[:2]/endpoint[2]
            distance = np.sum((normalized-xy)**2, axis=2)
            vv, uu = np.unravel_index(np.argmin(distance), distance.shape)
            # Do not turn an out-of-image projection into a misleading clipped arrow.
            if uu in (0,w-1) or vv in (0,h-1) or np.hypot(uu-u,vv-v) < 3:
                continue
            directions[name] = [int(uu),int(vv)]
            cv2.arrowedLine(overlay,(u,v),(int(uu),int(vv)),color,2,tipLength=.2)
            cv2.putText(overlay,name,(int(uu)+3,int(vv)),cv2.FONT_HERSHEY_SIMPLEX,.42,color,1)
        anchors.append(dict(origin_px=[u,v],endpoints_px=directions))
    if not any('-X ROBOT' in a['endpoints_px'] for a in anchors):
        raise ValueError('左腕机器人方向投影不可用；未发送缺少方向依据的分割请求')
    return overlay, dict(source='same_frame_calibrated_base_axes',anchors=anchors,
                        robot_direction='base -X',away_from_robot='base +X',
                        note='arrows are 25mm local displacements, not book boundaries')


def edge_evidence(depth, rays, transform, cover):
    """Describe every visible cover edge for a same-target corrective request."""
    cover = np.asarray(cover,float)
    inside = np.zeros(depth.shape,np.uint8)
    cv2.fillPoly(inside,[np.rint(cover).astype(np.int32)],1)
    inside = cv2.erode(inside,np.ones((3,3),np.uint8)).astype(bool)
    v,u = np.indices(depth.shape);uv=np.stack([u,v],axis=-1)
    rows=[]
    for i,a in enumerate(cover):
        b=cover[(i+1)%len(cover)];d=b-a;n=float(d@d)
        if n<1:continue
        t=(uv-a)@d/n
        mask=(inside & (t>=.1) & (t<=.9)
              & (np.linalg.norm(uv-(a+t[...,None]*d),axis=-1)<=5.)
              & np.isfinite(depth) & (depth>0) & (depth<=5000))
        row=dict(edge_index=i,endpoints_px=[a.tolist(),b.tolist()],samples=int(mask.sum()))
        if mask.sum()>=30:
            xyz=rays[mask]*(depth[mask,None].astype(float)*.001)
            xyz=(xyz@transform[:3,:3].T+transform[:3,3])*1000
            row.update(base_xyz_median_mm=np.median(xyz,axis=0).tolist(),
                       base_y_p5_p95_mm=np.percentile(xyz[:,1],[5,95]).tolist())
        rows.append(row)
    return rows


def selection_issue(spine_xyz, far_xyz):
    """Detect swapped side faces without tightening the existing contact margin."""
    s=np.percentile(spine_xyz[:,1],[5,95]);f=np.percentile(far_xyz[:,1],[5,95])
    if max(s[0],f[0]) > min(s[1],f[1]):
        return dict(reason='robot_facing_edges_have_disjoint_base_y',
                    spine_y_span_mm=s.tolist(),far_edge_y_span_mm=f.tolist())
    if np.median(far_xyz[:,0]) <= np.median(spine_xyz[:,0]):
        return dict(reason='far_edge_not_beyond_robot_facing_edge_in_base_x',
                    spine_x_mm=float(np.median(spine_xyz[:,0])),
                    far_edge_x_mm=float(np.median(far_xyz[:,0])))
    return None


def same_cover(first, second, shape):
    masks=[]
    for polygon in (first,second):
        mask=np.zeros(shape[:2],np.uint8)
        cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],1)
        masks.append(mask.astype(bool))
    union=np.count_nonzero(masks[0]|masks[1])
    return bool(union and np.count_nonzero(masks[0]&masks[1])/union>=.5)
