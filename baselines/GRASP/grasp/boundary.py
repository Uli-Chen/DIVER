"""Keep every generated file under this implementation; deny network by default."""
import os, sys, socket, shutil
from pathlib import Path
from .io import ROOT, inside


def install(online=False):
    os.chdir(ROOT)
    sys.dont_write_bytecode=True
    for key in ['TMPDIR','TMP','TEMP','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_DATA_HOME','MPLCONFIGDIR',
                'TIKTOKEN_CACHE_DIR','HF_HOME','TORCH_HOME','NUMBA_CACHE_DIR','UV_CACHE_DIR','NLTK_DATA']:
        p=ROOT/'runtime'/key.lower();p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',ANONYMIZED_TELEMETRY='false',HF_HUB_DISABLE_TELEMETRY='1',
        LITELLM_LOCAL_MODEL_COST_MAP='True',TOKENIZERS_PARALLELISM='false')
    import tempfile
    tempfile.tempdir=os.environ['TMPDIR']
    def audit(event,args):
        if event=='open':
            path,mode,flags=args
            if isinstance(path,(str,bytes,os.PathLike)) and (flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                inside(os.fsdecode(path))
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.chown','os.utime','os.truncate'}:
            if isinstance(args[0],(str,bytes,os.PathLike)):inside(os.fsdecode(args[0]))
            else:raise PermissionError('Descriptor-based mutation unsupported')
        elif event in {'os.rename','os.replace'}:
            inside(os.fsdecode(args[0]));inside(os.fsdecode(args[1]))
        elif event in {'os.symlink','os.link'}:raise PermissionError('Generated links forbidden')
        elif event in {'socket.connect','socket.getaddrinfo','socket.sendto'} and not online:
            raise PermissionError('Model/network execution is disabled pending user audit')
        elif event=='subprocess.Popen':
            argv=args[1]
            if not isinstance(argv,(list,tuple)) or not argv or str(argv[0]) not in ['/usr/bin/sandbox-exec', os.environ.get('RMUX', shutil.which('rmux') or '/opt/homebrew/bin/rmux')]:
                raise PermissionError('Unconfined subprocess forbidden')
    sys.addaudithook(audit)


def kernel(online=False):
    if sys.platform!='darwin':raise RuntimeError('This adapter requires the configured macOS sandbox')
    import ctypes,json
    lib=ctypes.CDLL('/usr/lib/libsandbox.dylib')
    lib.sandbox_init.argtypes=[ctypes.c_char_p,ctypes.c_uint64,ctypes.POINTER(ctypes.c_char_p)]
    profile='(version 1) (allow default) (deny file-write*) (allow file-write* (subpath '+json.dumps(str(ROOT))+'))'
    if not online:profile+=' (deny network*)'
    error=ctypes.c_char_p()
    if lib.sandbox_init(profile.encode(),0,ctypes.byref(error)):raise RuntimeError('OS sandbox could not be installed')
