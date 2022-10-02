Note: To run this program simply place the files without the videos in a folder named dataset
# likeabosch2022
Bosch: Software Development Challenge.
Challenge topic: Object tracking and blind spot detection
The problem of object tracking and blind spot is a multifaceted one, It requires numerous amount of steps before we can get reasonable results. Over the course of the challenge, I experimented and settled on a solution. A solution that both track an object giving each object an ID in real-time requiring only frame t-1 and t.
My model was built as a combination of the hungarian algorithm (linear_sum_assignment)  for object tracking and identification and 2D kalman filter for position estimation.

The Hungarian matching algorithm, also called the Kuhn-Munkres algorithm, is a O(∣V∣3)O(∣V∣ 3) algorithm that can be used to find maximum-weight matchings in bipartite graphs, which is sometimes called the assignment problem. I framed the frame to frame object tracking as a matching algorithm modelling the cost with an l2-norm, the l2-norm was used because this is a good measure of distance in a euclidean plane. After further investigation a more proper choice of cost was modelled, but not implemented at the current momment. The modelled cost was to use the velocity as part of the cost for this problem.
 Nevertheless, the algorithm works, but a better solution would be to use feature representation from pre-trained autoencoders or convolutional filters trained on large dataset to compute a siemens lost to use as the new cost for our hungarian algorithm, and coupled with motion tracking, we would moving closer to the current state of the art in object tracking.
 
The second part of my model architecture was the kalman filter, this is a well known technique for estimating object states using probabilities. It has been used extensively and studied, so I wont go indept on it. Although a better model using more state of the art technique would be preferred, like modelling the interaction of every object in the scene as a graph node and their geometric relationship would be modelled as the graph edges. This method has achieve state of the art in varied field ranging from medicine - alphafold 2-  to core engineering -modelling physics simulation- and even to understand human behaviour.
My proposal is to explore this techniques and other gradient based model optimization, Interactions in complex scenes fits so rightly with GNN-graph neural network architecture and with recent surge in large language models, where the key property can be seen as its ability to find meaningful representational features to encode text, this way allowing it to make more coherent predictions. The aim is to drive the vision community into this paradigm, which we have already seen with DETR-detection transformer and ViT-vision transformer.

BreakDown of Code

 The first step with any meaningful computer vision or machine learning project is to first analyze and clean your data. With this I studied the pdf provided to us to understanding how the data are represented. After scaling and preprocessing, I seperated the data into meaning list as required for my project. I converted the data provided for the vehicle motion into x,y coordinate in image plain using basic kinematic eqn.
x_2 = x_1 + rotated_velocity_using_angle * change_time

angle = initial_angle + degree(angular_velocity) * change_time
Then, I animated the motion of the vehicle. Also made label for the track number.
