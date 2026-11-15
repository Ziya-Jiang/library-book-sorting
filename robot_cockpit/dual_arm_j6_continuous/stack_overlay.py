"""Per-book visible regions from the original GPT frame; display only."""
import cv2
import numpy as np

COLORS=((0,220,255),(255,150,40),(140,245,90),(230,100,255),(255,100,140),(100,160,255))

def decode_books(books,count,rect,*,pixel_coordinates=False):
    if not isinstance(books,list) or len(books)!=count:
        raise ValueError('逐本标注数量与总本数不一致；不生成虚构编号区域')
    x,y,x1,y1=rect;out=[]
    for number,item in enumerate(books,1):
        if not isinstance(item,dict) or type(item.get('number')) is not int or item['number']!=number:
            raise ValueError('逐本编号须从顶书1开始连续排列')
        try:points=np.asarray(item.get('visible_polygon'),float)
        except (TypeError,ValueError):raise ValueError('逐本可见区域坐标无效') from None
        if (points.ndim!=2 or points.shape[1:]!=(2,) or not 3<=len(points)<=12
            or not np.isfinite(points).all() or np.any(points<0)):
            raise ValueError('逐本可见区域须为3至12个有效图像坐标')
        if pixel_coordinates:
            if np.any(points>=[x1-x,y1-y]):raise ValueError('左腕逐本区域坐标超出图像')
            native=np.rint(points).astype(int)+[x,y]
        else:
            if np.any(points>1000):raise ValueError('逐本归一化区域超出图像')
            native=np.rint(points*[(x1-x-1)/1000,(y1-y-1)/1000]+[x,y]).astype(int)
        out.append(dict(number=number,visible_polygon_px=native.tolist()))
    return out

def draw_books(rgb,books,count,lift_mm):
    image=rgb.copy();h,w=image.shape[:2]
    for i,book in enumerate(books):
        points=np.asarray(book['visible_polygon_px'],np.int32)
        color=COLORS[i%len(COLORS)]
        mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[points],1)
        on=mask.astype(bool);image[on]=(.78*image[on]+.22*np.array(color)).astype(np.uint8)
        cv2.polylines(image,[points],True,color,2,cv2.LINE_AA)
        center=tuple(np.rint(points.mean(axis=0)).astype(int))
        row=int(52+i*max(15,min(26,(h-60)/max(1,len(books)))))
        anchor=(8,min(h-5,row))
        cv2.line(image,(70,anchor[1]-4),center,color,1,cv2.LINE_AA)
        cv2.putText(image,f'Book {book["number"]}',anchor,cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(image,f'Book {book["number"]}',anchor,cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
    cv2.rectangle(image,(0,0),(w-1,29),(20,20,20),-1)
    cv2.putText(image,f'Books: {count}   Lift: ({count}+1)cm = {lift_mm:g}mm',(8,21),
                cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return image
