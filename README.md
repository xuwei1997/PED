# Image Segmentation with Contrastive Learning for Plant Time-series Images with Priori Distance Embedding
## Introduction

Current plant phenotyping studies have focused extensively on plant time-series image studies utilizing deep learning. Such study images are easier to obtain but costly to annotate, and contrast learning is one method for efficient label training. The growth of plants is slow, their image sequences change little over time, and their semantic information is simple. Previous contrast pre-training models struggle to distinguish between positive samples from different enhanced views of the same image and similar negative samples from different images. For this reason, this paper proposes a contrastive learning method with a priori distance embedding (PDE) for plant time-series images. Different phenological stages of plants correspond to different semantic information in the images. The method converts this essential domain knowledge into a priori distances between image pairs and performs comparative learning pre-training; the weights can then be transferred to the semantic segmentation task. Based on this method, the results of experiments on time-series images of cherries demonstrate that the PDE comparative learning method can be effectively applied to the pre-training of semantic segmentation of plant time-series images, and has a wide range of potential applications in computer vision plant phenotypic research.


![img_8.png](img_8.png)
![img_2.png](img_2.png)
![img_3.png](img_3.png)
![img_4.png](img_4.png)

## Requirement
- python = 3.6
- tensorflow = 1.8.0
- Keras = 2.1.6

## Result

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=610
 style='width:457.8pt;border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:14.0pt'>
  <td width=171 nowrap valign=top style='width:128.05pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'></td>
  <td width=205 nowrap valign=top style='width:153.75pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'></td>
  <td width=117 nowrap colspan=2 valign=top style='width:88.0pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Frozen<o:p></o:p></span></p>
  </td>
  <td width=117 nowrap colspan=2 valign=top style='width:88.0pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Fine-tuning<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:14.0pt'>
  <td width=171 nowrap valign=top style='width:128.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'></td>
  <td width=205 nowrap valign=top style='width:153.75pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Pre-training Data Set<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>mIoU</span></span><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>mPA</span></span><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>mIoU</span></span><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>mPA</span></span><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:14.0pt'>
  <td width=171 nowrap valign=top style='width:128.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Glorot</span></span><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'> Uniform<o:p></o:p></span></p>
  </td>
  <td width=205 nowrap valign=top style='width:153.75pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'></td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.2309<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.3183<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.2652<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.3758<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:14.0pt'>
  <td width=376 nowrap colspan=2 valign=top style='width:281.8pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Glorot</span></span><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'> Uniform (from scratch)<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'></td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'></td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.2719<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.3666<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:14.0pt'>
  <td width=171 nowrap valign=top style='width:128.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>SimCLR</span></span><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=205 nowrap valign=top style='width:153.75pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>7373 cherry images<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.0733<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.2<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.2678<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.382<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:14.0pt'>
  <td width=171 nowrap valign=top style='width:128.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MoCo</span></span><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=205 nowrap valign=top style='width:153.75pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>7373 cherry images<o:p></o:p></span></p>
  </td>
  <td width=235 nowrap colspan=4 valign=top style='width:176.0pt;border-top:
  none;border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Pre-training non convergence<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:14.0pt'>
  <td width=171 nowrap valign=top style='width:128.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>SimSiam</span></span><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=205 nowrap valign=top style='width:153.75pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>7373 cherry images<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.2252<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.3146<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.2609<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.347<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:14.0pt'>
  <td width=171 nowrap valign=top style='width:128.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>ImageNet<o:p></o:p></span></p>
  </td>
  <td width=205 nowrap valign=top style='width:153.75pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>More than 10000000 images<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.2969<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.4158<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.5412<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.716<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;height:14.0pt'>
  <td width=171 nowrap valign=top style='width:128.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Supervised phenological
  classification<o:p></o:p></span></p>
  </td>
  <td width=205 nowrap valign=top style='width:153.75pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3100 cherry images<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.343<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.4502<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.3883<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.498<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9;mso-yfti-lastrow:yes;height:14.0pt'>
  <td width=171 nowrap valign=top style='width:128.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>PDE (classification
  distance)<o:p></o:p></span></p>
  </td>
  <td width=205 nowrap valign=top style='width:153.75pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=left style='text-align:left;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>600000 image pairs from 3100
  cherry images<o:p></o:p></span></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><b><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.5502<o:p></o:p></span></b></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><b><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.7123<o:p></o:p></span></b></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><b><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.5751<o:p></o:p></span></b></p>
  </td>
  <td width=59 nowrap valign=top style='width:44.0pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:14.0pt'>
  <p class=MsoNormal align=right style='text-align:right;mso-pagination:widow-orphan'><b><span
  lang=EN-US style='font-size:11.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.7355<o:p></o:p></span></b></p>
  </td>
 </tr>
</table>

## Figure

![img_5.png](img_5.png)
![img_6.png](img_6.png)
![img_7.png](img_7.png)

## dir
1.