"""Launcher tests; no detector or interactive viewer is launched here."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import launch_viewers as L


class Tests(unittest.TestCase):
    def test_all_entry_paths_and_modes(self):
        with patch.object(L,'runner',return_value=['python']):
            for view in L.VIEWS:
                for mode in ('auto','step'):
                    cmd=L.command(view,mode)
                    self.assertTrue(Path(cmd[1]).is_file())
                    self.assertEqual(cmd[cmd.index('--mode')+1],mode)
                    self.assertNotIn('--no-eval',cmd)
                    self.assertNotIn('--score-profile',cmd)

    def test_three_four_panels_and_five_presets(self):
        with patch.object(L,'runner',return_value=['python']):
            self.assertTrue(L.command('three')[1].endswith('final_three_cam_no_band.py'))
            self.assertTrue(L.command('three-band')[1].endswith('final_three_cam.py'))
            cmd=L.command('five'); self.assertEqual(cmd[cmd.index('--panorama')+1],'natural')
            self.assertNotIn('--dashboard',L.command('surround'))

    def test_spatial_alias_and_dedicated_bev_page(self):
        with patch.object(L,'runner',return_value=['python']):
            self.assertEqual(L.command('spatial'),L.command('bev'))
            self.assertTrue(L.command('spatial')[1].endswith('final_spatial_cam.py'))
            cmd=L.command('bev-radar',start='30',headless=True,limit=1)
            self.assertTrue(cmd[1].endswith('final_immersive_cam.py'))
            self.assertEqual(cmd[cmd.index('--page')+1],'bev')
            self.assertEqual(cmd[cmd.index('--start')+1],'00000030')
            self.assertIn('--headless',cmd)
            self.assertEqual(cmd[cmd.index('--limit')+1],'1')
            self.assertNotIn('--dashboard',cmd)

    def test_frame_normalization_and_spaces_not_shell(self):
        with patch.object(L,'runner',return_value=['python']):
            cmd=L.command('three',start='30',out=Path('folder with spaces'))
            self.assertEqual(cmd[cmd.index('--start')+1],'00000030')
            self.assertEqual(cmd[cmd.index('--out')+1],str(Path('folder with spaces').resolve()))
            self.assertIsInstance(cmd,list)

    def test_bad_inputs_rejected(self):
        for kw in (dict(view='missing'),dict(view='five',cache=True),dict(view='three',start='1 & echo x'),
                   dict(view='three',fps=float('nan')),dict(view='three',fps=0),dict(view='three',limit=-1),
                   dict(view='three',mode='step',headless=True),dict(view='three',start='123456789')):
            with self.assertRaises(ValueError):L.command(**kw)

    def test_child_root_and_no_shell(self):
        with patch.object(subprocess,'Popen') as mocked:
            cmd=['python','a file.py'];L.start_viewer(cmd)
            args,kw=mocked.call_args
            self.assertEqual(args,(cmd,));self.assertEqual(kw['cwd'],L.ROOT)
            self.assertFalse(kw.get('shell',False))

    def test_uv_and_offline_fallback(self):
        with patch.object(L.shutil,'which',return_value='uv'):
            r=L.runner();self.assertIn('--offline',r);self.assertIn('--no-sync',r)
        with patch.object(L.shutil,'which',return_value=None):
            self.assertTrue(Path(L.runner()[0]).is_file())

    def test_tk_menu_layout_and_selection(self):
        import tkinter as tk
        root=tk.Tk();root.withdraw()
        try:
            app=L.Launcher(root);root.update_idletasks()
            self.assertEqual(app.selected(),'surround')
            self.assertIn('等比例',app.help.get())
            app.start.set('30');app.fps.set('2')
            app.quick_buttons['bev'].invoke()
            self.assertEqual(app.selected(),'bev')
            self.assertIn('小屏分页',app.help.get())
            app.quick_buttons['bev-radar'].invoke()
            self.assertEqual(app.selected(),'bev-radar')
            self.assertIn('大画面BEV',app.help.get())
            self.assertEqual(app.start.get(),'30');self.assertEqual(app.fps.get(),'2')
            self.assertIsNone(app.child)
            app.quick_buttons['surround'].invoke()
            self.assertEqual(app.selected(),'surround')
            app.cache.set(True);app.combo.current(app.keys.index('five'));app.select()
            self.assertFalse(app.cache.get());self.assertEqual(str(app.cache_button.cget('state')),'disabled')
            self.assertLessEqual(root.winfo_reqwidth(),700)
            self.assertLessEqual(root.winfo_reqheight(),650)
        finally:root.destroy()


if __name__=='__main__':unittest.main()
