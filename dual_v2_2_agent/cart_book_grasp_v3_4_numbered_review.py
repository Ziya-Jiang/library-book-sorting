"""Review optional head tilt without changing the operator's numbered book.

Only the selected instance is visible to GPT. Its mask remains the identity
boundary and the whole-book measured depth source; no automatic reselection.
"""
import numpy as np
from cart_book_grasp_v3_head_tilt_review import review
from cart_book_grasp_v3_2_head_fallback import recover


def review_locked_spine(candidate,rgb,depth,intrinsic,c2w,*,checkpoint=lambda:None,
                       observation_robot=None,decision=None,config=None,fallback=None,center=None):
    identity=str(candidate['id'])
    book=np.asarray(candidate['book_mask'],bool).copy()
    if book.shape!=depth.shape or rgb.shape[:2]!=book.shape or not book.any():
        raise ValueError('编号复核缺少有效的原书本区域')
    original=candidate['geometry']
    isolated=np.zeros_like(rgb)
    isolated[book]=rgb[book]
    audit=dict(selection_id=identity,number=candidate.get('number'),
               scene_id=(decision or {}).get('scene_id'),
               identity_policy='selected_instance_only',gpt_called=False)

    def locked_recover(image,detector,*,checkpoint,config,review_context):
        audit['gpt_called']=True
        if decision is not None:
            decision['gpt_called']=True
            decision['head_tilt_review']=audit
        context=dict(review_context,locked_selection=audit.copy(),
                     masked_background=True)
        detections,_=(fallback or recover)(isolated,None,checkpoint=checkpoint,
                                           config=config,review_context=context)
        if len(detections)!=1:
            raise ValueError('编号复核必须返回选中书的唯一书脊；不更换目标')
        score,mask,_=detections[0]
        mask=np.asarray(mask,bool)
        if mask.shape!=book.shape:
            raise ValueError('编号复核图像尺寸不匹配')
        locked=mask&book
        if not locked.any():
            raise ValueError('GPT复核未落在选中编号书内；不更换目标')
        audit.update(returned_pixels=int(mask.sum()),locked_pixels=int(locked.sum()))
        y,x=np.nonzero(locked)
        return [(score,locked,[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)])],None

    try:
        selected,_=review(original,isolated,depth,intrinsic,c2w,
            checkpoint=checkpoint,config=config,fallback=locked_recover,center=center,
            observation_robot=observation_robot)
        if selected is original:
            return candidate
        # A GPT whole-book polygon must never change the numbered instance's
        # measured X endpoint. Only the reviewed spine/tilt is replaced.
        selected=dict(selected)
        for key,value in original.items():
            if key.startswith('whole_book_') or key in (
                    'book_surface_x_mm','selection_box_xyxy','segmentation_crop_xyxy'):
                selected[key]=value
        selected['head_tilt_review']['locked_selection']=audit.copy()
        candidate['geometry']=selected
        if decision is not None:
            decision['head_tilt_review']=selected['head_tilt_review']
        return candidate
    except BaseException as error:
        audit['error']=f'{type(error).__name__}: {error}'
        raise
