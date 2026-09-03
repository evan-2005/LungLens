"""
Launch the LungLens Gradio app for the Fig. 6 screenshots, serving the VERIFIED
classifier - the git-HEAD chest_model_4class.pth, which reproduces
chest_classifier_metrics.json exactly (96.1757% test, macro-F1 95.7709%).
The working-tree checkpoint is a different, weaker model (91.20%).

The switch is made by chdir-ing into fig7_work/serve_root/ BEFORE importing app:
app.py resolves chest_model_4class.pth, chest_segmentation_model.pth and
chest_classifier_metrics.json relative to the working directory, so the normal
startup path loads the right files and the header and metrics panel report them
correctly. Patching the globals after import would leave the header, which is
rendered while the Blocks are built, saying "No model loaded".

allowed_paths exposes the read-only kagglehub cache so the capture script can
hand the real test file to the upload widget byte-for-byte, rather than a
re-encoded copy that would change the prediction.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(os.path.join(HERE, "serve_root"))
sys.path.insert(0, ROOT)

import app

print(f"[serve] cls={app.cls_model is not None} seg={app.seg_model is not None} "
      f"seg_disease={app.seg_disease_model} device={app.device} cwd={os.getcwd()}",
      flush=True)

KAG = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub")
app.demo.queue().launch(server_name="127.0.0.1", server_port=7861, share=False,
                        css=app.ll_css, allowed_paths=[KAG])
