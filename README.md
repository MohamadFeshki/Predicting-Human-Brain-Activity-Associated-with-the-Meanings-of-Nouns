# Predicting-Human-Brain-Activity-Associated-with-the-Meanings-of-Nouns
This program is generated to analyze the results of the source paper: https://www.ncbi.nlm.nih.gov/pubmed/18511683

Data are provided in http://science.sciencemag.org/content/320/5880/1191/tab-figures-data

For the fast version, using "Coefficient1.mat" it would be possible to bypass the linear fitting step and just use its coefficients.

Extra .mat files should be added in the workspace before running the program. These files are listed below:

val .mat: The concrete nouns list
verbarenged.mat: The semantic feature list
Fw.mat: Weights of a trillion-word text corpus 
Coefficient1.mat: linear regression coefficients are saved in this file, for the fast mode.
