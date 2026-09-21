#!/usr/bin/env python3
import sys
sys.dont_write_bytecode=True
if __name__=='__main__':
    try:
        from grasp.cli import main
        main()
    except Exception as error:
        # Provider exceptions can contain response text or secrets; emit class/frames only.
        import json,traceback
        print(json.dumps({'error_type':type(error).__name__,
            'frames':[{'file':f.filename,'line':f.lineno,'function':f.name} for f in traceback.extract_tb(error.__traceback__)]}),file=sys.stderr)
        raise SystemExit(2)
