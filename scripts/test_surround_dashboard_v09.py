"""Aspect, resize and mouse regression, independent of interactive windows."""
import sys
import unittest
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'ultralytics/decided')]
import final_five_cam
from immersive_surround import ImmersiveSphere,LookController,overview
from immersive_dashboard import SurroundDashboard
from multiview_geometry import load_cameras
import cv2
import numpy as np


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sphere=ImmersiveSphere(load_cameras(),384,192)
        cls.texture=np.zeros((192,384,3),np.uint8)
        cls.images={c:np.zeros((50,80,3),np.uint8) for c in range(1,6)}
        cls.bev=np.zeros((100,100,3),np.uint8)

    def test_sphere_has_exact_original_aspect(self):
        for w in (640,1000,1280):
            result=overview(self.texture,self.sphere,LookController(),w)
            self.assertEqual(result.shape,(25+w//2,w,3))

    def test_pages_fill_canvas_at_small_and_fullscreen_sizes(self):
        ui=SurroundDashboard(self.sphere);c=LookController()
        for w,h in ((800,600),(1280,720),(1920,1080)):
            for page in ('look','sphere','cameras','bev'):
                result=ui.render(self.texture,c,self.images,[],self.bev,'test',dict(precision=0,recall=0,f1=0),1,w,h,page)
                self.assertEqual(result.shape,(h,w,3))
                if page=='look':
                    self.assertEqual((c.width,c.height),(w,h-120))
                    self.assertEqual((ui.viewport.width,ui.viewport.height),(c.width,c.height))
                self.assertEqual(ui.hit(20,60),ord('m'))

    def test_nonviewport_and_inset_do_not_rotate(self):
        ui=SurroundDashboard(self.sphere);c=LookController()
        def draw(page='look',context=False):
            return ui.render(self.texture,c,self.images,[],self.bev,'test',dict(precision=0,recall=0,f1=0),1,1280,720,page,context)
        draw();ui.mouse(c,cv2.EVENT_LBUTTONDOWN,200,200);ui.mouse(c,cv2.EVENT_MOUSEMOVE,300,200)
        self.assertNotEqual(c.yaw,0)
        old=c.yaw;ui.mouse(c,cv2.EVENT_MOUSEMOVE,300,60);ui.mouse(c,cv2.EVENT_MOUSEMOVE,400,200)
        self.assertEqual(c.yaw,old)
        draw(context=True);x,y,w,h=ui.inset
        ui.mouse(c,cv2.EVENT_LBUTTONDOWN,x+3,y+3);self.assertIsNone(c.drag)
        draw('sphere');ui.mouse(c,cv2.EVENT_LBUTTONDOWN,200,200);self.assertIsNone(c.drag)


if __name__=='__main__':unittest.main()
