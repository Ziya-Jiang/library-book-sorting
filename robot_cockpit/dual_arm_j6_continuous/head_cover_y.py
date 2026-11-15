"""J6-only head Y measurement using temporally stable visible-cover interior."""
import cv2
import numpy as np
from cover_region import top_mask
from flat_book_push.right_trial import approach_target

INSET_PX=5
MIN_POINTS=30


def wrist_spine_y(reference,batch_id,epoch,position_mm,wxyz):
    """Use this round's already validated wrist RGB-D spine, never a side/cover fallback."""
    if (reference.get('batch_id')!=batch_id or not batch_id or reference.get('epoch')!=epoch
            or not reference.get('frame_token')
            or reference.get('method')!='initial_left_wrist_spine_base_Z_median'
            or reference.get('sample_count',0)<MIN_POINTS):
        raise ValueError('缺少同轮同会话的有效左腕书脊深度')
    center=np.asarray(reference.get('center_base_mm'),float)
    if center.shape!=(3,) or not np.isfinite(center).all():
        raise ValueError('左腕书脊基座坐标无效')
    target=np.asarray(position_mm,float).copy();target[1]=center[1]
    return dict(mode='approach_only',
                measured=dict(method='median_measured_base_y_in_left_wrist_robot_facing_spine',
                              reference='robot_facing_spine',center_mm=center.tolist(),
                              source_frame_token=reference['frame_token'],
                              sample_count=reference['sample_count']),
                offset_y_mm=float(target[1]-position_mm[1]),
                stages=[dict(stage='approach_y',translation_axis='Y',
                             tcp_m_wxyz=np.r_[target*.001,wxyz].tolist())])


def measure_spine_y(frames,polygon,diagnostics,position_mm,wxyz):
    """Measure only the independently identified robot-facing spine."""
    mask=top_mask(frames[-1].depth_mm.shape,polygon)
    inner=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),
                   borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    if inner.sum()<MIN_POINTS:
        raise ValueError('朝机器人书脊内侧不足30个测深像素；不回退到封面或相机侧面')
    diagnostics.update(sampling_region='robot_facing_spine_interior',inset_px=1,
                       selected_pixels=int(mask.sum()),interior_pixels=int(inner.sum()),
                       minimum_stable_points=MIN_POINTS)
    result=approach_target(frames,inner,diagnostics,position_mm,wxyz,require_spatial_coverage=False)
    result['measured'].update(method='median_measured_base_y_in_robot_facing_spine',
                              reference='robot_facing_spine',inset_px=1)
    print(f"J6朝机器人书脊测Y：跨帧稳定{diagnostics['stable_depth_pixels']}点，"
          f"书脊Y={result['measured']['center_mm'][1]:.3f}mm，offset={result['offset_y_mm']:+.3f}mm。",flush=True)
    return result


def interior_mask(shape,polygon):
    cover=top_mask(shape,polygon)
    kernel=np.ones((2*INSET_PX+1,2*INSET_PX+1),np.uint8)
    inner=cv2.erode(cover.astype(np.uint8),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    if inner.sum()<MIN_POINTS:
        raise ValueError(f'可见封面向内收缩{INSET_PX}px后不足{MIN_POINTS}个测深像素；不回退到边缘')
    return cover,inner


def measure_cover_y(frames,polygon,diagnostics,position_mm,wxyz,*,partial=False):
    cover,inner=interior_mask(frames[-1].depth_mm.shape,polygon)
    counts=[]
    for frame in frames:
        if frame.depth_mm.shape!=inner.shape:raise ValueError('测量期间深度图尺寸改变')
        z=frame.depth_mm[inner]
        counts.append(int((np.isfinite(z)&(z>0)&(z<=5000)).sum()))
    diagnostics.update(sampling_region='visible_cover_interior',inset_px=INSET_PX,
                       visible_cover_pixels=int(cover.sum()),interior_pixels=int(inner.sum()),
                       per_frame_valid_pixels=counts,
                       per_frame_valid_fraction=[n/float(inner.sum()) for n in counts],
                       partial_visible_cover=bool(partial),minimum_stable_points=MIN_POINTS)
    result=approach_target(frames,inner,diagnostics,position_mm,wxyz,require_spatial_coverage=False)
    result['measured'].update(method='median_measured_base_y_in_stable_visible_cover_interior',
                              full_book_center_observed=not partial,
                              partial_visible_cover=bool(partial),inset_px=INSET_PX)
    print(f"J6封面内侧测Y：内缩{INSET_PX}px，选区{inner.sum()}点，跨帧稳定{diagnostics['stable_depth_pixels']}点，"
          f"基座Y={result['measured']['center_mm'][1]:.3f}mm。",flush=True)
    return result
