# The codes of Improved-undamaged-to-damaged-acceleration-response-translation-for-Structural-Health-Monitoring

Published in Elsevier's Engineering Applications of Artificial Intelligence: https://www.sciencedirect.com/science/article/pii/S0952197623003305

Unpaired image-to-image translation is a popular research topic in computer vision and graphics. Recently, the authors of this paper took a similar approach and translated the domain of acceleration responses collected from a steel grandstand structure. In doing so, the undamaged response is translated to damaged, and the damaged response to undamaged. For that, a variant of the CycleGAN model is trained with undamaged (bolt tightened at the joint) and damaged (bolt loosened at the joint) responses from a single joint in the structure. However, the success of the domain translation on the test joints was very limited. 

Thus, this study investigates improvements to the model and the training procedure for further accuracy. First, the model in this study gets a more extensive training procedure to increase the model’s domain knowledge. During the training, a novel signal coherence-based index is considered to account for the similarity of frequency domains of the original and the translated data. Second, the Gated Linear Units, skip-connections, and Mish activation function are used to minimize the gradient loss and to learn the broader features in the data. Third, the total loss function of the generator is supplemented with a new frequency domain-based loss to better capture the frequency content of the data. Fourth, random decaying noise is added to the inputs for better generalization in the test data. Last, the model is evaluated using modal parameters such as natural frequencies, damping ratios, and singular value decomposition of the estimated spectral densities. The improvements presented in this study demonstrate a successful domain translation of acceleration responses for the tested joints compared to past study. The findings of this paper show that domain translation can be advantageous in Structural Health Monitoring applications, such as having access to the damaged or undamaged response of the structure while it is in pristine or unhealthy condition.


# Codes
config.py provides the configurations used in the training

critic.py provides the critic model

generator.py provides the generator model and the other blocks used in both critic and generator

metrics.py provides some of the metrics used in the training

train.py is the file for training the generators and critics

utils.py is only used for gradient penalty used for the critics during the training

# Single model architecture of the CycleWDCGAN-GP model
![1-s2 0-S0952197623003305-gr4_lrg](https://github.com/furknluleci/Improved-undamaged-to-damaged-acceleration-response-translation-for-Structural-Health-Monitoring/assets/63553991/a7f92629-6552-40a0-a850-67f6942dd66d)

# Training data flow for (a) undamaged-to-damaged domain translation and (b) damaged-to-undamaged domain translation
![1-s2 0-S0952197623003305-gr5_lrg](https://github.com/furknluleci/Improved-undamaged-to-damaged-acceleration-response-translation-for-Structural-Health-Monitoring/assets/63553991/190057c9-4dd3-45cd-942c-08087a56dfaf)

# Some of the Results (see the modal identification results in the paper)
The frequency domains of the concatenated response signals are plotted and shown in Fig. 8, Fig. 9, Fig. 10, Fig. 11, Fig. 12, Fig. 13. For instance, Fig. 8(a) shows the frequency domain plotting of undamaged and synthetic undamaged acceleration response signals at Joint 5; on the other hand, Fig. 8(b) shows the frequency domain plotting of damaged and synthetic damaged acceleration response signals at Joint 5. For modal identification results, please see the original paper https://www.sciencedirect.com/science/article/pii/S0952197623003305
![Picture1](https://github.com/furknluleci/Improved-undamaged-to-damaged-acceleration-response-translation-for-Structural-Health-Monitoring/assets/63553991/c5843dd1-fdc2-493a-9269-e330e8bea6b5)
