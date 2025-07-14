# MeshNet4carvings
MeshNet modified for prehistoric carving recognition

To deploy the MeshNet carving recognition method on a rock carving site requires that you are (i) looking for more examples of a preexisting carving motifs, (ii) there are multiple examples (e.g. more than 50) of the preexisting carving motifs and that you have access to a dataset of their 3-D representation, and (iii) there is little physical overlap between instances of each existing motif. If these conditions are met, MeshNet can be employed using the following steps:

1.	Gather a structure from motion photogrammetric dataset of the preexisting carving motifs and convert them into mesh representation.

2.	Preprocess the meshes by firstly importing them into MeshLab (https://www.meshlab.net/). Crop out each individual carving motif using a rectangular mask. For each carving motif, also crop out a similar-sized rectangular region of the stone surface that does not feature the carving motif. The result should be an equal number of carving samples to non-carving samples. Then, simplify each mesh to 1024 faces using the ‘Quadric Edge Collapse Decimation’ filter in MeshLab.

3.	Import the simplified meshes into Blender and standardise both carving and non-carving meshes by centring: aligning the midpoint of each mesh, rotating: aligning the average outward normal of each mesh with the positive z-axis, and scaling: ensuring all meshes are to-scale with its real-life counterpart. Using the ‘preprocess.py’ file convert all meshes into the ‘.npz’ file format, which is required for input to MeshNet. These meshes are considered the training dataset.

4.	Train MeshNet on the prepared training dataset. Explore the MeshNet hyperparameters to obtain a trained MeshNet that performs with a high F-score. Begin by setting the maximum epoch to 100 and batch size to 40, and then use a grid search to explore the values 0.0005, 0.001, 0.002, and 0.003 for both learning rate and weight decay. However, if this produces low F-scores, expand the search of all hyperparameters to larger and smaller values. Ensure MeshNet is not overfitted to the training data by plotting and monitoring the loss and accuracy.

5.	To apply the trained MeshNet to the rock surface of interest will require further research into dividing the surface into separate samples like the cropping of each carving motif performed in step 2. To perform an extensive search over the surface requires combining methods such as the sliding 3-D detection window [1] and a 3-D variant of the image pyramid [2].

6.	Test the trained MeshNet on the test dataset. The accuracy of MeshNet on the test dataset should be like that for the training dataset.

[1] S. Song, J. Xiao, Sliding Shapes for 3D Object Detection in Depth Images, in: Comput. Vision–ECCV 2014 13th Eur. Conf., Zurich, 2014: pp. 634–651. https://doi.org/10.1007/978-3-319-10599-4_41.
[2]	P.F. Felzenszwalb, R.B. Girshick, D. McAllester, D. Ramanan, Object Detection with Discriminatively Trained Part-Based Models, IEEE Trans. Pattern Anal. Mach. Intell. 32 (2010) 1627–1645. https://doi.org/10.1109/TPAMI.2009.167.
