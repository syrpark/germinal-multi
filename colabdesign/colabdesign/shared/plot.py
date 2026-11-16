# import matplotlib
import numpy as np
from scipy.special import expit as sigmoid
from colabdesign.shared.protein import _np_kabsch, alphabet_list

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib.gridspec import GridSpec 
from matplotlib import collections as mcoll
try:
  import py3Dmol
except:
  print("py3Dmol not installed")
  
pymol_color_list = ["#33ff33","#00ffff","#ff33cc","#ffff00","#ff9999","#e5e5e5","#7f7fff","#ff7f00",
                    "#7fff7f","#199999","#ff007f","#ffdd5e","#8c3f99","#b2b2b2","#007fff","#c4b200",
                    "#8cb266","#00bfbf","#b27f7f","#fcd1a5","#ff7f7f","#ffbfdd","#7fffff","#ffff7f",
                    "#00ff7f","#337fcc","#d8337f","#bfff3f","#ff7fff","#d8d8ff","#3fffbf","#b78c4c",
                    "#339933","#66b2b2","#ba8c84","#84bf00","#b24c66","#7f7f7f","#3f3fa5","#a5512b"]

jalview_color_list = {"Clustal":           ["#80a0f0","#f01505","#00ff00","#c048c0","#f08080","#00ff00","#c048c0","#f09048","#15a4a4","#80a0f0","#80a0f0","#f01505","#80a0f0","#80a0f0","#ffff00","#00ff00","#00ff00","#80a0f0","#15a4a4","#80a0f0"],
                      "Zappo":             ["#ffafaf","#6464ff","#00ff00","#ff0000","#ffff00","#00ff00","#ff0000","#ff00ff","#6464ff","#ffafaf","#ffafaf","#6464ff","#ffafaf","#ffc800","#ff00ff","#00ff00","#00ff00","#ffc800","#ffc800","#ffafaf"],
                      "Taylor":            ["#ccff00","#0000ff","#cc00ff","#ff0000","#ffff00","#ff00cc","#ff0066","#ff9900","#0066ff","#66ff00","#33ff00","#6600ff","#00ff00","#00ff66","#ffcc00","#ff3300","#ff6600","#00ccff","#00ffcc","#99ff00"],
                      "Hydrophobicity":    ["#ad0052","#0000ff","#0c00f3","#0c00f3","#c2003d","#0c00f3","#0c00f3","#6a0095","#1500ea","#ff0000","#ea0015","#0000ff","#b0004f","#cb0034","#4600b9","#5e00a1","#61009e","#5b00a4","#4f00b0","#f60009","#0c00f3","#680097","#0c00f3"],
                      "Helix Propensity":  ["#e718e7","#6f906f","#1be41b","#778877","#23dc23","#926d92","#ff00ff","#00ff00","#758a75","#8a758a","#ae51ae","#a05fa0","#ef10ef","#986798","#00ff00","#36c936","#47b847","#8a758a","#21de21","#857a85","#49b649","#758a75","#c936c9"],
                      "Strand Propensity": ["#5858a7","#6b6b94","#64649b","#2121de","#9d9d62","#8c8c73","#0000ff","#4949b6","#60609f","#ecec13","#b2b24d","#4747b8","#82827d","#c2c23d","#2323dc","#4949b6","#9d9d62","#c0c03f","#d3d32c","#ffff00","#4343bc","#797986","#4747b8"],
                      "Turn Propensity":   ["#2cd3d3","#708f8f","#ff0000","#e81717","#a85757","#3fc0c0","#778888","#ff0000","#708f8f","#00ffff","#1ce3e3","#7e8181","#1ee1e1","#1ee1e1","#f60909","#e11e1e","#738c8c","#738c8c","#9d6262","#07f8f8","#f30c0c","#7c8383","#5ba4a4"],
                      "Buried Index":      ["#00a35c","#00fc03","#00eb14","#00eb14","#0000ff","#00f10e","#00f10e","#009d62","#00d52a","#0054ab","#007b84","#00ff00","#009768","#008778","#00e01f","#00d52a","#00db24","#00a857","#00e619","#005fa0","#00eb14","#00b649","#00f10e"]}

pymol_cmap = matplotlib.colors.ListedColormap(pymol_color_list)
    
def show_pdb(pdb_str, show_sidechains=False, show_mainchains=False,
             color="pLDDT", chains=None, Ls=None, vmin=50, vmax=90,
             color_HP=False, size=(800,480), hbondCutoff=4.0):
  
  if chains is None:
    chains = 1 if Ls is None else len(Ls)

  view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', width=size[0], height=size[1])
  view.addModel(pdb_str,'pdb',{'hbondCutoff':hbondCutoff})
  if color == "pLDDT":
    view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':vmin,'max':vmax}}})
  elif color == "rainbow":
    view.setStyle({'cartoon': {'color':'spectrum'}})
  elif color == "chain":
    for n,chain,color in zip(range(chains),alphabet_list,pymol_color_list):
       view.setStyle({'chain':chain},{'cartoon': {'color':color}})
  if show_sidechains:
    BB = ['C','O','N']
    HP = ["ALA","GLY","VAL","ILE","LEU","PHE","MET","PRO","TRP","CYS","TYR"]
    if color_HP:
      view.addStyle({'and':[{'resn':HP},{'atom':BB,'invert':True}]},
                    {'stick':{'colorscheme':"yellowCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':HP,'invert':True},{'atom':BB,'invert':True}]},
                    {'stick':{'colorscheme':"whiteCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                    {'sphere':{'colorscheme':"yellowCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                    {'stick':{'colorscheme':"yellowCarbon",'radius':0.3}})
    else:
      view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                    {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                    {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
      view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                    {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})  
  if show_mainchains:
    BB = ['C','O','N','CA']
    view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
  view.zoomTo()
  return view

def plot_pseudo_3D(xyz, c=None, ax=None, chainbreak=5, Ls=None,
                   cmap="gist_rainbow", line_w=2.0,
                   cmin=None, cmax=None, zmin=None, zmax=None,
                   shadow=0.95):

  def rescale(a, amin=None, amax=None):
    a = np.copy(a)
    if amin is None: amin = a.min()
    if amax is None: amax = a.max()
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin)/(amax - amin)

  # make segments and colors for each segment
  xyz = np.asarray(xyz)
  if Ls is None:
    seg = np.concatenate([xyz[:,None],np.roll(xyz,1,0)[:,None]],axis=1)
    c_seg = np.arange(len(seg))[::-1] if c is None else (c + np.roll(c,1,0))/2
  else:
    Ln = 0
    seg = []
    c_seg = []
    for L in Ls:
      sub_xyz = xyz[Ln:Ln+L]
      seg.append(np.concatenate([sub_xyz[:,None],np.roll(sub_xyz,1,0)[:,None]],axis=1))
      if c is not None:
        sub_c = c[Ln:Ln+L]
        c_seg.append((sub_c + np.roll(sub_c,1,0))/2)
      Ln += L
    seg = np.concatenate(seg,0)
    c_seg = np.arange(len(seg))[::-1] if c is None else np.concatenate(c_seg,0)
  
  # set colors
  c_seg = rescale(c_seg,cmin,cmax)  
  if isinstance(cmap, str):
    if cmap == "gist_rainbow": 
      c_seg *= 0.75
    colors = matplotlib.cm.get_cmap(cmap)(c_seg)
  else:
    colors = cmap(c_seg)
  
  # remove segments that aren't connected
  seg_len = np.sqrt(np.square(seg[:,0] - seg[:,1]).sum(-1))
  if chainbreak is not None:
    idx = seg_len < chainbreak
    seg = seg[idx]
    seg_len = seg_len[idx]
    colors = colors[idx]

  seg_mid = seg.mean(1)
  seg_xy = seg[...,:2]
  seg_z = seg[...,2].mean(-1)
  order = seg_z.argsort()

  # add shade/tint based on z-dimension
  z = rescale(seg_z,zmin,zmax)[:,None]

  # add shadow (make lines darker if they are behind other lines)
  seg_len_cutoff = (seg_len[:,None] + seg_len[None,:]) / 2
  seg_mid_z = seg_mid[:,2]
  seg_mid_dist = np.sqrt(np.square(seg_mid[:,None] - seg_mid[None,:]).sum(-1))
  shadow_mask = sigmoid(seg_len_cutoff * 2.0 - seg_mid_dist) * (seg_mid_z[:,None] < seg_mid_z[None,:])
  np.fill_diagonal(shadow_mask,0.0)
  shadow_mask = shadow ** shadow_mask.sum(-1,keepdims=True)

  seg_mid_xz = seg_mid[:,:2]
  seg_mid_xydist = np.sqrt(np.square(seg_mid_xz[:,None] - seg_mid_xz[None,:]).sum(-1))
  tint_mask = sigmoid(seg_len_cutoff/2 - seg_mid_xydist) * (seg_mid_z[:,None] < seg_mid_z[None,:])
  np.fill_diagonal(tint_mask,0.0)
  tint_mask = 1 - tint_mask.max(-1,keepdims=True)

  colors[:,:3] = colors[:,:3] + (1 - colors[:,:3]) * (0.50 * z + 0.50 * tint_mask) / 3
  colors[:,:3] = colors[:,:3] * (0.20 + 0.25 * z + 0.55 * shadow_mask)

  set_lim = False
  if ax is None:
    fig, ax = plt.subplots()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    set_lim = True
  else:
    fig = ax.get_figure()
    if ax.get_xlim() == (0,1):
      set_lim = True
      
  if set_lim:
    xy_min = xyz[:,:2].min() - line_w
    xy_max = xyz[:,:2].max() + line_w
    ax.set_xlim(xy_min,xy_max)
    ax.set_ylim(xy_min,xy_max)

  ax.set_aspect('equal')
    
  # determine linewidths
  width = fig.bbox_inches.width * ax.get_position().width
  linewidths = line_w * 72 * width / np.diff(ax.get_xlim())

  lines = mcoll.LineCollection(seg_xy[order], colors=colors[order], linewidths=linewidths,
                               path_effects=[matplotlib.patheffects.Stroke(capstyle="round")])
  
  return ax.add_collection(lines)

def plot_ticks(ax, Ls, Ln=None, add_yticks=False):
  if Ln is None: Ln = sum(Ls)
  L_prev = 0
  for L_i in Ls[:-1]:
    L = L_prev + L_i
    L_prev += L_i
    ax.plot([0,Ln],[L,L],color="black")
    ax.plot([L,L],[0,Ln],color="black")
  
  if add_yticks:
    ticks = np.cumsum([0]+Ls)
    ticks = (ticks[1:] + ticks[:-1])/2
    ax.yticks(ticks,alphabet_list[:len(ticks)])

