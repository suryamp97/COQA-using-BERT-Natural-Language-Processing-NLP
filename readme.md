Project Details emphasized in the paper.

1) python main_Script.py
main_script.py contains the Bert model and the training part. After the execution
a directory named Bert would be created which will contain the predictions file

2) python evaluate-v1.0 --data-file data/coqa-dev-v1.0.json --pred-file Bert/predictions.json
After the training is complete and the predictions are made, run the evaluate-v1.0 file 
(downloaded from https://stanfordnlp.github.io/coqa/)
with the above arguments to get the results. 


Files:
Main_script.py => Contains the Bert model and the training part.
evaluate-v1.0  => official CoQA evaluation script
Data
=> coqa-dev-v1.0.json (Development dataset)
=> coqa-train-v1.0 (Training dataset)
=>processors
	=> metrics.py (For computing the predictions from the output logits)
	=> coqa.py (Pre-processing)
	=> utils.py (Utility file that supplements pre-processing)





