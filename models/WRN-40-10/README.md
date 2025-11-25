# WRN-40-10 Experiment

WRN-40-10 was trained with the following hyperparameters:

* Number of Server Rounds (`num-server-rounds`): 300
* Local Client Epochs (`local-epochs`): 4
* Learning Rate (`lr`): 0.05
* Fraction of Clients Sampled for Training (`fraction-train`): 0.75

Here are the final results after training:

* Train Loss(`train_loss`): 0.9346144306333625
* Evaluation Loss (`eval_loss`): 1.187270419293234
* Evaluation Accuracy (`eval_acc`): 0.7273879142300195

Here are the results of the MIA:

* Accuracy: 0.7170020598954207
* Precision: 0.724766125061546
* Recall: 0.6997306290603708
* F1 score: 0.7120283779425992

The purpose of documenting this attack is to show that the MIA I run is successful. I want to show that this MIA is successful because in future models, training on an architecture as large as WRN-40-10 with differential privacy implemented is not possible in the Google Colab Environment. Therefore, we must use a smaller architecture, and that architecture may underfit the model. Demonstrating a successful MIA highlights the threat it poses, and the need for a mitigation like differential privacy.
