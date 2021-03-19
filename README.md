this code trains an NLP LSTM on a dataset (netflix_titles.csv) of synopsis and descriptions/generes from Netflix shows. The resulting model is
multilabel and can predict the top most likely descriptions/genres for any given any given synopsis. The novelty of this code is that it uses
Tensorflow's TextVectorization layer, which converts each word into a numeric vector. This vectors are an embedded layer of the model and 
change as it trains, such that similar words will result in similar vectors.

Example of the model predicting the top five genres for the given synopsis:

the synopsis is:
  
Before planning an awesome wedding for his grandfather, a polar bear king must take back a stolen artifact from an evil archaeologist first.

the top five predicted genres are:

comedies, children & family movies, romantic tv shows, action & adventure, international tv shows
