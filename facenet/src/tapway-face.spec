# -*- mode: python -*-

block_cipher = None

a = Analysis(['tapway-face.py'],
             pathex=['/home/oka96/Desktop/5-4-tapway/facenet/src'],
             binaries=[],
             datas=[('/home/oka96/Desktop/5-4-tapway/facenet/src/align/*.npy','align'),
                    ('/home/oka96/Desktop/5-4-tapway/facenet/src/gui/icon.jpg','gui'),
                    ('/home/oka96/Desktop/5-4-tapway/facenet/src/config.json','.')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='tapway-face',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
