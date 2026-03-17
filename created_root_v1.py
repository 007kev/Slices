#created_root_v1

import uproot
f = uproot.open("kev_Pppim_eFD_006665.root")
tree = f["Individual"]
tree.keys()
