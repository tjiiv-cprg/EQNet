import os

if eval(os.environ.get("USE_MEMEFF_ATT", "False")):
    print("[INFO] Memory Efficient FFN Attention is enabled.")
    from qcnet_plug_in.memeff_att import AttentionLayer
else:
    from .attention_layer import AttentionLayer
